from typing import Any, List
import math 

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
    ):
        super().__init__()
        self.save_hyperparameters()
        self.test_acc = Accuracy()
        self.val_acc = Accuracy()
        self.train_acc = Accuracy()
        self.model = build_model(config, num_classes=config.data.num_classes, data_shape=config.data.shape)  # TODO: change cifar_stem later for more other datasets
        self.k = self.hparams.config.active.k

    def training_step(self, batch, batch_idx):
        mode = "train"
        loss, preds, y = self.step(batch, k=1)
        self.log(f"{mode}/loss", loss)
        self.train_acc.update(preds, y)
        return loss

    def validation_step(self, batch, batch_idx):
        mode = "val"
        loss, preds, y = self.step(batch)
        self.log(f"{mode}/loss", loss, on_step=False, on_epoch=True)
        self.val_acc.update(preds, y)
        return loss

    def test_step(self, batch, batch_idx):
        mode = "test"
        loss, preds, y = self.step(batch)
        self.log(f"{mode}/loss", loss, on_step=False, on_epoch=True)
        self.test_acc.update(preds, y)
        return loss

    def on_train_epoch_start(self) -> None:
        self.train_acc.reset()

    def on_validation_epoch_start(self) -> None:
        self.val_acc.reset()

    def on_test_epoch_start(self) -> None:
        self.test_acc.reset()

    def on_train_epoch_end(self, *args) -> None:
        mode = "train"
        self.log(f"{mode}/acc", self.train_acc.compute(), on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        mode = "val"
        self.log(f"{mode}/acc", self.val_acc.compute(), on_step=False, on_epoch=True)

    def on_test_epoch_end(self):
        mode = "test"
        self.log(f"{mode}/acc", self.test_acc.compute(), on_step=False, on_epoch=True)


    def step(self, batch: Any, k:int=None):
        if k is None:
            k = self.k
        x, y = batch
        logits = self.forward(x, k=k)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def configure_optimizers(self):
        optimizer_name = self.hparams.config.optim.optimizer.name
        lr = self.hparams.config.model.learning_rate
        wd = self.hparams.config.model.weight_decay
        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                params=self.parameters(),
                lr=lr,
                weight_decay=wd,
            )
        elif optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                params=self.parameters(),
                lr=lr,
                weight_decay=wd,
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
            step_size = self.hparams.config.trainer.max_epochs // 4
            if step_size == 0:
                step_size+=1
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer,
                step_size=step_size,
            )
            return [optimizer], [scheduler]
        else:
            raise NotImplementedError

    def forward(self, x: torch.Tensor, k: int = None, agg: bool = True):
        if k is None:
            k = self.k
        out = self.model.forward(x, k)  # B x k x ....
        if agg:
            if k == 1:
                return torch.log_softmax(out.squeeze(1), dim=-1)
            else: 
                out = torch.log_softmax(out, dim=-1)
                out = torch.logsumexp(out, dim=1) - math.log(k)
                return out
        return out
