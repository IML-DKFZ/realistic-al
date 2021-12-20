from abc import abstractclassmethod
import math
from urllib.parse import non_hierarchical

import torch
import pytorch_lightning as pl
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay
from torchmetrics import Accuracy
from copy import deepcopy
from typing import Tuple
from torch.nn import functional as F

from .utils import exclude_from_wt_decay, freeze_layers, load_from_ssl_checkpoint
from .callbacks.ema_callback import EMAWeightUpdate


class AbstractClassifier(pl.LightningModule):
    def __init__(
        self,
        eman: bool = True,
    ):
        super().__init__()

        # general model
        self.train_iters_per_epoch = None

        self.ema_model = None
        self.eman = eman

        self.acc_train = Accuracy()
        self.acc_val = Accuracy()
        self.acc_test = Accuracy()

    def forward(
        self, x: torch.Tensor, k: int = None, agg: bool = True, ema: bool = False
    ):
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

    def mc_nll(self, logits: torch.Tensor):
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

    def get_features(self, x: torch.Tensor):
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
        logits = self.forward(x, k=k)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        mode = "val"
        loss, preds, y = self.step(batch)
        self.log(f"{mode}/loss", loss, on_step=False, on_epoch=True)
        self.acc_val.update(preds, y)

    def test_step(self, batch, batch_idx, *args, **kwargs):
        mode = "test"
        loss, preds, y = self.step(batch)
        self.log(f"{mode}/loss", loss, on_step=False, on_epoch=True)
        self.acc_test.update(preds, y)

    def on_train_batch_end(
        self, outputs, batch, batch_idx: int, dataloader_idx: int
    ) -> None:
        if self.ema_model is not None:
            self.ema_weight_update.on_train_batch_end(
                self.trainer, self, outputs, batch, batch_idx, dataloader_idx
            )

    def on_train_epoch_start(self) -> None:
        self.acc_train.reset()
        if self.hparams.model.freeze_encoder:
            self.model.resnet.eval()
        # When ema model is used during training, correct buffers should be used
        # e.g. eman fixmatch should use eman-batchnorm for teacher!
        if self.ema_model is not None:
            self.ema_model.eval()

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

    def setup(self, *args, **kwargs) -> None:
        self.train_iters_per_epoch = len(self.train_dataloader())

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
            params_cf = exclude_from_wt_decay(
                self.model.classifier.named_parameters(),
                weight_decay=wd,
                skip_list=no_decay,
                learning_rate=0.1,
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
                # lr=lr,
                # weight_decay=wd,
            )
        elif optimizer_name == "sgd":
            momentum = self.hparams.optim.optimizer.momentum
            nesterov = self.hparams.optim.optimizer.nesterov
            optimizer = torch.optim.SGD(
                params,
                # lr=lr, weight_decay=wd,
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

    # TODO: Change this so that it works for many different models!
    def load_from_ssl_checkpoint(self):
        """Loads a Self-Supervised Resnet from a Checkpoint obtained from PL Bolts"""
        load_from_ssl_checkpoint(self.model, path=self.hparams.model.load_pretrained)

    def wrap_dm(self, dm: pl.LightningDataModule) -> pl.LightningDataModule:
        return dm
