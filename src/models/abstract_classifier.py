from abc import abstractclassmethod
import math
import numpy as np
from omegaconf import DictConfig

import torch
import pytorch_lightning as pl
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay
from torchmetrics import Accuracy
from copy import deepcopy
from typing import Tuple, Union
from torch.nn import functional as F
import torch.nn as nn
import torchvision
from pathlib import Path
from loguru import logger
from pytorch_lightning.loggers import TensorBoardLogger


from .utils import exclude_from_wt_decay, freeze_layers, load_from_ssl_checkpoint
from .callbacks.ema_callback import EMAWeightUpdate


class AbstractClassifier(pl.LightningModule):
    def __init__(self, eman: bool = True):
        """Abstract Classifier carrying the logic for Bayesian Models with MC Dropout and logging for base values.
        Dropout is per default used always, also during validation due to nature of Bayesian Model (Yarin Gal)
        """
        super().__init__()

        # general model
        self.train_iters_per_epoch = None

        self.ema_model = None
        self.eman = eman

        self.acc_train = Accuracy()
        self.acc_val = Accuracy()
        self.acc_test = Accuracy()

        self.loss_fct = nn.NLLLoss()

    def forward(
        self, x: torch.Tensor, k: int = None, agg: bool = True, ema: bool = False
    ) -> torch.Tensor:
        """Forward Pass which selects the correct model and returns class logprobabilities if agg=True,
        else it returns the logits"""
        model_forward = self.select_forward_model(ema=ema)
        if k is None:
            k = self.k

        out = model_forward(x, k)  # B x k x ....

        if k == 1 and len(out.shape) == 3:
            out = out.squeeze(1)

        if agg:
            out = self.mc_nll(out)
        return out

    def mc_nll(self, logits: torch.Tensor) -> torch.Tensor:
        out = torch.log_softmax(logits, dim=-1)
        if len(logits.shape) > 2:
            k = out.shape[1]
            out = torch.logsumexp(out, dim=1) - math.log(k)
        return out

    @abstractclassmethod
    def training_step(self, *args, **kwargs) -> torch.Tensor:
        """Training Step which logs training accuracy and returns loss"""
        return super().training_step(*args, **kwargs)

    def select_forward_model(self, ema: bool = False) -> torch.nn.Module:
        """Selects exponential moving avg or normal model according to training state"""
        if ema and self.ema_model is not None:
            return self.ema_model
        elif self.training:
            return self.model
        else:
            if self.ema_model is not None:
                return self.ema_model
            else:
                return self.model

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        model_forward = self.select_forward_model()
        return model_forward.get_features(x).flatten(start_dim=1)

    def init_ema_model(self, use_ema: bool = False):
        if use_ema:
            self.ema_model = deepcopy(self.model)
            for param in self.ema_model.parameters():
                param.requires_grad = False
            self.ema_weight_update = EMAWeightUpdate(eman=self.eman)

    def step(self, batch: Tuple[torch.tensor, torch.tensor], k: int = None):
        x, y = batch
        logprob = self.forward(x, k=k)
        loss = self.loss_fct(logprob, y)
        # loss = F.nll_loss(logprob, y)
        preds = torch.argmax(logprob, dim=1)
        return loss, logprob, preds, y

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        mode = "val"
        loss, logprob, preds, y = self.step(batch)
        self.log(f"{mode}/loss", loss, on_step=False, on_epoch=True)
        self.acc_val.update(preds, y)
        if batch_idx == 0 and self.current_epoch == 0:
            if len(batch[0].shape) == 4:
                self.visualize_inputs(batch[0], name=f"{mode}/data")
        return logprob, y

    def test_step(self, batch, batch_idx, *args, **kwargs):
        mode = "test"
        loss, logprob, preds, y = self.step(batch)
        self.log(f"{mode}/loss", loss, on_step=False, on_epoch=True)
        self.acc_test.update(preds, y)
        if batch_idx == 0:
            if len(batch[0].shape) == 4:
                self.visualize_inputs(batch[0], name=f"{mode}/data")
        return logprob, y

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        if self.ema_model is not None:
            self.ema_weight_update.on_train_batch_end(
                self.trainer, self, outputs, batch, batch_idx
            )

    def on_train_epoch_start(self) -> None:
        self.acc_train.reset()
        if self.hparams.model.freeze_encoder:
            self.model.resnet.eval()
        # When ema model is used during training, correct buffers should be used
        # e.g. eman fixmatch should use eman-batchnorm for teacher!
        if self.ema_model is not None:
            self.ema_model.eval()

    def on_fit_start(self) -> None:
        """Initialize metrics for the tensorboard_logger.
        Either self.loggers is a list with tb_logger as first,
        or only tb_logger is present."""
        metric_placeholder = {"val/acc": 0.0, "test/acc": 0.0}
        for logger in self.loggers:
            if isinstance(logger, TensorBoardLogger):
                logger.log_hyperparams(self.hparams, metrics=metric_placeholder)

    def on_validation_epoch_start(self) -> None:
        self.acc_val.reset()

    def on_test_epoch_start(self) -> None:
        self.acc_test.reset()

    def on_train_epoch_end(self) -> None:
        mode = "train"
        self.log(f"{mode}/acc", self.acc_train.compute(), on_step=False, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        mode = "val"
        self.log(f"{mode}/acc", self.acc_val.compute(), on_step=False, on_epoch=True)

    def on_test_epoch_end(self) -> None:
        mode = "test"
        self.log(f"{mode}/acc", self.acc_test.compute(), on_step=False, on_epoch=True)

    def setup_data_params(self, dm: pl.LightningDataModule):
        """Create internal parameter with the amount of training iterations per epoch.

        Args:
            dm (pl.LightningDataModule): DataModule
        """
        train_loader = dm.train_dataloader()
        if isinstance(train_loader, (tuple, list)):
            self.train_iters_per_epoch = max([len(loader) for loader in train_loader])
        else:
            self.train_iters_per_epoch = len(train_loader)

        # This implementation is correct and uses the amount of labels from the train_loader.
        # Therefore if Resampling is used, more samples are used.
        weighted_loss = False
        if "weighted_loss" in self.hparams.model:
            weighted_loss: bool = self.hparams.model.weighted_loss
        if weighted_loss:
            logger.info("Initializing Weighted Loss")
            if hasattr(dm.train_set, "targets"):
                classes: np.ndarray = dm.train_set.targets
            else:
                # workaround for FixMatch trainings with multiple dataloaders

                if isinstance(train_loader, (tuple, list)):
                    # train_loader 0 is the labeled loader
                    train_loader = train_loader[0]
                classes = []
                for x, y in train_loader:
                    classes.append(y.numpy())
                classes = np.concatenate(classes)

            classes, class_weights = np.unique(classes, return_counts=True)
            # computation identical to sklearn balanced class weights
            # https://github.com/scikit-learn/scikit-learn/blob/36958fb24/sklearn/utils/class_weight.py#L10
            class_weights = torch.tensor(
                np.sum(class_weights) / (len(classes) * class_weights),
                dtype=torch.float,
            )
            self.loss_fct = nn.NLLLoss(weight=class_weights)

    def configure_optimizers(self):
        optimizer_name = self.hparams.optim.optimizer.name
        lr = self.hparams.model.learning_rate
        wd = self.hparams.model.weight_decay
        exclude_bn_bias = self.hparams.model.exclude_bn_bias
        if self.hparams.model.freeze_encoder:
            freeze_layers(self.model.resnet)
        if exclude_bn_bias:
            no_decay = ["bn", "bias"]
        else:
            no_decay = []
        if self.hparams.model.finetune:
            # Additional Finetuing with different LRs for CF and Encoder
            # Benefits according to https://arxiv.org/abs/2101.08482
            params_enc = exclude_from_wt_decay(
                self.model.resnet.named_parameters(),
                weight_decay=wd,
                skip_list=no_decay,
                learning_rate=lr,
            )
            # LR = 0.1 in SelfMatch but 0.03 works better in my setting with 40 labels on cifar10
            params_cf = exclude_from_wt_decay(
                self.model.classifier.named_parameters(),
                weight_decay=wd,
                skip_list=no_decay,
                learning_rate=0.03,
            )
            params = params_enc + params_cf
        else:
            params = exclude_from_wt_decay(
                self.model.named_parameters(),
                weight_decay=wd,
                skip_list=no_decay,
                learning_rate=lr,
            )
        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                params,
            )
        elif optimizer_name == "sgd":
            momentum = self.hparams.optim.optimizer.momentum
            nesterov = self.hparams.optim.optimizer.nesterov
            optimizer = torch.optim.SGD(
                params,
                momentum=momentum,
                nesterov=nesterov,
            )
        else:
            raise NotImplementedError

        scheduler_name = self.hparams.optim.lr_scheduler.name
        if scheduler_name is None:
            return [optimizer]
        elif scheduler_name == "cosine_decay":
            train_epochs_per_iter = self.train_iters_per_epoch

            warm_steps = (
                self.hparams.optim.lr_scheduler.warmup_epochs * train_epochs_per_iter
            )
            max_steps = self.hparams.trainer.max_epochs * train_epochs_per_iter
            lin_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=optimizer,
                lr_lambda=linear_warmup_decay(warm_steps, max_steps, cosine=True),
            )

            scheduler = {
                "scheduler": lin_scheduler,
                "interval": "step",
                "frequency": 1,
            }
            return [optimizer], [scheduler]
        elif scheduler_name == "steplr":
            step_size = self.hparams.trainer.max_epochs // 4
            if step_size == 0:
                step_size += 1
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer,
                step_size=step_size,
            )
            return [optimizer], [scheduler]
        elif scheduler_name == "steplr_resnet":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=[160], gamma=0.1
            )
            return [optimizer], [scheduler]
        else:
            raise NotImplementedError

    def visualize_inputs(self, inputs, name):
        num_imgs = 64
        num_rows = 8
        grid = (
            torchvision.utils.make_grid(
                inputs[:num_imgs], nrow=num_rows, normalize=True
            )
            .cpu()
            .detach()
        )
        if len(self.loggers) > 0:
            self.loggers[0].experiment.add_image(
                name,
                grid,
                self.current_epoch,
            )

    # TODO: Change this so that it works for many different models!
    def load_from_ssl_checkpoint(self):
        """Loads a Self-Supervised Resnet from a Checkpoint obtained from PL Bolts"""
        load_from_ssl_checkpoint(self.model, path=self.hparams.model.load_pretrained)

    def wrap_dm(self, dm: pl.LightningDataModule) -> pl.LightningDataModule:
        return dm

    def load_only_state_dict(self, path):
        ckpt = torch.load(path)
        logger.debug("Loading Model from Path: {}".format(path))
        logger.info("Loading checkpoint from Epoch: {}".format(ckpt["epoch"]))
        self.load_state_dict(ckpt["state_dict"], strict=True)

    def get_best_ckpt(
        self, experiment_path: Union[str, Path], use_last: bool = True
    ) -> Path:
        """Return the path to the best checkpoint

        Args:
            experiment_path (Union[str, Path]): path to base experiment

        Returns:
            Path: Best checkpoint path
        """
        model_ckpt_path = Path(experiment_path) / "checkpoints"
        ckpts = [ckpt for ckpt in model_ckpt_path.iterdir() if ckpt.suffix == ".ckpt"]
        # print(ckpts)
        if "last.ckpt" in [ckpt.name for ckpt in ckpts] and use_last:
            model_ckpt = model_ckpt_path / "last.ckpt"
        else:
            ckpts_f = [ckpt for ckpt in ckpts if "last.ckpt" not in ckpt.name]
            ckpts_f.sort(key=lambda x: x.name.split("=")[1].split("-")[0])
            if len(ckpts_f) == 0:
                raise FileNotFoundError(
                    "Path {} has no checkpoints ".format(model_ckpt_path)
                )
            model_ckpt = ckpts_f[-1]
        return model_ckpt
