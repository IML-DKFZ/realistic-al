from typing import Any, List, Optional, Tuple
import math
from omegaconf import DictConfig

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from data.data import TorchVisionDM
from models.networks import build_model
from .abstract_classifier import AbstractClassifier

from data.sem_sl import wrap_fixmatch_train_dataloader


class FixMatch(AbstractClassifier):
    def __init__(
        self, config: DictConfig,
    ):
        """FixMatch Classifier, which can be extended to a bayesian Neural network by setting config.dropout_p to values greater 0."""
        super().__init__(eman=config.sem_sl.eman)
        # maybe add possiblity to use resnext/use
        self.save_hyperparameters(config)
        self.model = build_model(
            config, num_classes=config.data.num_classes, data_shape=config.data.shape
        )

        # semi supervised
        self.cf_thresh = config.sem_sl.cf_thresh
        self.lambda_u = config.sem_sl.lambda_u
        self.T_semsl = config.sem_sl.T
        self.mu = config.sem_sl.mu

        self.k = self.hparams.model.k
        if self.hparams.model.load_pretrained:
            self.load_from_ssl_checkpoint()
        self.init_ema_model(use_ema=config.model.use_ema)

        self.balanced_loss = False
        weighted_loss = False
        try:
            weighted_loss = self.hparams.model.weighted_loss
        except:
            pass
        if weighted_loss:
            self.balanced_loss = True
            buffer_size = 128
            num_classes = self.hparams.data.num_classes
            self.loss_fct = nn.NLLLoss(weight=torch.ones(num_classes))
            self.register_buffer(
                "p_model", torch.ones(buffer_size, num_classes) / num_classes
            )
            # p_models is PMovingAVG according to: https://github.com/google-research/remixmatch/blob/f7061ebf055227cbeb5c6fced1ce054e0ceecfcd/mixmatch.py#L80
            # PMovingAverage: https://github.com/google-research/remixmatch/blob/f7061ebf055227cbeb5c6fced1ce054e0ceecfcd/libml/layers.py#L125

            # goal is to obtain a balanced classifer, therefore p_data is uniform
            self.register_buffer("p_data", torch.ones(num_classes) / num_classes)

    def training_step(self, batch, batch_idx, *args, **kwargs):
        mode = "train"
        labeled_batch, unlabeled_batch = batch
        x, y = labeled_batch
        (x_w, x_s), _ = unlabeled_batch
        logits, logits_w, logits_s = self.obtain_logits(x, x_w, x_s)
        loss_u, mask_u = self.semi_step(logits_w, logits_s)
        loss_s, preds, y = self.step_logits(logits, y)
        loss = loss_s + self.lambda_u * loss_u
        self.log(
            f"{mode}/loss_s",
            loss_s,
            on_step=False,
            on_epoch=True,
            batch_size=x.shape[0],
        )
        self.log(
            f"{mode}/loss_u",
            loss_u,
            on_step=False,
            on_epoch=True,
            batch_size=x.shape[0],
        )
        # define batchsize for mean computation due to ambiguity
        self.log(
            f"{mode}/loss", loss, on_step=False, on_epoch=True, batch_size=x.shape[0]
        )
        self.log(
            f"{mode}/mask_u",
            mask_u,
            on_step=False,
            on_epoch=True,
            batch_size=x_w.shape[0],
        )
        # compute estimate for accuracy additionally to mask
        # self.log(f"{mode}/mask_acc", logits.argmax())
        self.acc_train.update(preds, y)
        # only save very first batch of first epoch to minimize loading times and memory footprint
        if batch_idx == 0 and self.current_epoch == 0:
            if len(x.shape) == 4:
                self.visualize_train(x, x_w, x_s)
        return {
            "loss": loss,
            "logprob": F.softmax(logits, dim=-1),
            "label": y,
        }

    def step_logits(self, logits, y):
        preds = torch.argmax(logits, dim=1)
        loss = self.loss_fct(F.log_softmax(logits, dim=-1), y)
        return loss, preds, y

    def obtain_logits(self, x, x_w, x_s):
        if self.eman and self.ema_model is not None:
            with torch.no_grad():
                logits_w = self.forward(x_w, k=None, agg=False, ema=True)
            x_full = torch.cat([x, x_s])
            x_full = interleave(x_full, x_full.shape[0])
            logits_ = self.forward(x_full, k=1, agg=False)
            logits_ = de_interleave(logits_, logits_.shape[0])
            logits = logits_[: x.shape[0]]
            logits_s = logits_[x.shape[0] :]
        else:
            x_full = torch.cat([x, x_w, x_s])
            x_full = interleave(x_full, x_full.shape[0])
            logits_ = self.forward(x_full, k=1, agg=False)
            logits_ = de_interleave(logits_, logits_.shape[0])
            logits = logits_[: x.shape[0]]
            logits_w, logits_s = logits_[x.shape[0] :].chunk(2)
        return logits, logits_w, logits_s

    def semi_step(self, logits_w, logits_s):
        # create mask for temp scaled output probabilities greater equal threshold
        probs = torch.exp(self.mc_nll(logits_w.detach() / self.T_semsl))
        if self.balanced_loss:
            # Distribution Alignment according to:
            # https://github.com/google-research/remixmatch/blob/master/remixmatch_no_cta.py#L36
            probs *= (1e-6 + self.p_data) / (1e-6 + self.get_p_model())
            probs /= probs.sum(dim=-1, keepdim=True)
        max_probs, pseudo_labels = torch.max(probs, dim=1)
        mask = max_probs.ge(self.cf_thresh).float()
        # weigh according to amount of samples used
        loss = (
            F.cross_entropy(logits_s, pseudo_labels, reduction="none") * mask
        ).mean()
        return loss, mask.mean()

    def update_p_model(self, prob: torch.Tensor):
        """Updates the self.p_model moving average with the given probabilites.

        Args:
            prob (torch.Tensor): Shape=NxC
        """
        self.p_model = torch.concat(
            [self.p_model[1:], prob.mean(dim=0, keepdim=True).detach()]
        )

    def get_p_model(self):
        """Returns Moving Average of p_model on the guessed labels.

        Returns:
            (torch.Tensor): Shape=C
        """
        p_model = self.p_model.mean(dim=0)
        p_model /= p_model.sum()
        return p_model

    def visualize_train(self, x, x_w, x_s):
        num_imgs = 64
        num_rows = 8
        for imgs, title in zip(
            [x, x_w, x_s], ["samples_lab", "samples_weak", "samples_strong"],
        ):
            if len(imgs) == 0:
                continue
            self.visualize_inputs(imgs, name=f"train/{title}")
            # grid = (
            #     torchvision.utils.make_grid(
            #         imgs[:num_imgs], nrow=num_rows, normalize=True
            #     )
            #     .cpu()
            #     .detach()
            # )
            # # Works only for Tensorboard!
            # self.loggers[0].experiment.add_image(
            #     title, grid, self.current_epoch,
            # )

    def setup(self, *args, **kwargs) -> None:
        # get_train_dataloader = wrap_fixmatch_train_dataloader(
        #     self.trainer.datamodule, self.mu
        # )
        # self.trainer.datamodule.train_dataloader = get_train_dataloader
        # self.train_dataloader = get_train_dataloader
        super().setup()

    def on_test_epoch_end(self) -> None:
        return super().on_test_epoch_end()

    def wrap_dm(self, dm: TorchVisionDM) -> TorchVisionDM:
        dm.train_dataloader = wrap_fixmatch_train_dataloader(dm, self.mu)
        return dm


def interleave(x, size):
    """"""
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])
