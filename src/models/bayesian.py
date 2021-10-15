from typing import Any, List

import torch
import pytorch_lightning as pl
from torchmetrics import Accuracy
import torch.nn.functional as F
from models.networks import build_model
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay
from pl_bolts.optimizers.lars import LARS


class BayesianModule(pl.LightningModule):
    def __init__(
        self,
        config,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.test_acc = Accuracy()
        self.val_acc = Accuracy()
        self.train_acc = Accuracy()
        self.model = build_model(config)

    def training_step(self, batch, batch_idx):
        mode = "train"
        loss, preds, y = self.step(batch)
        self.log(f"{mode}/loss", loss)
        self.train_acc.update(preds, y)
        self.log(f"{mode}/acc", self.train_acc.compute(), on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        mode = "val"
        loss, preds, y = self.step(batch)
        self.log(f"{mode}/loss", loss)
        self.val_acc.update(preds, y)
        self.log(f"{mode}/acc", self.val_acc.compute(), on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        mode = "test"
        loss, preds, y = self.step(batch)
        self.log(f"{mode}/loss", loss)
        self.val_acc.update(preds, y)
        self.log(f"{mode}/acc", self.val_acc.compute(), on_step=False, on_epoch=True)
        return loss

    def step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def configure_optimizers(self):
        optimizer_name = self.hparams.config.optim.optimizer.name
        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                params=self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )
        elif optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                params=self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            raise NotImplementedError

        scheduler_name = self.hparams.config.optim.lr_scheduler.name

        if scheduler_name is None:
            return [optimizer]

        elif scheduler_name == "cosine_decay":
            raise NotImplementedError(
                "Train Iterations per Epoch still needs to be Implemented!"
            )
            warm_steps = (
                self.hparams.optim.lr_scheduler.warmup_epochs
                * self.train_iters_per_epoch
            )
            max_steps = self.trainer.max_epochs * self.train_iters_per_epoch
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
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer,
                step_size=self.hparams.config.trainer.max_epochs // 4,
            )
            return [optimizer], [scheduler]
        else:
            raise NotImplementedError

    def forward(self, x: torch.Tensor, k: int = 1):
        out = self.model.forward(x, k)  # B x k x ....
        if k == 1:
            return out.squeeze(1)
        return out
