from typing import Any, List

import torch
import pytorch_lightning as pl
from torchmetrics import Accuracy
import torch.nn.functional as F
from models.networks import build_model


class BayesianModule(pl.LightningModule):
    def __init__(
        self,
        config,
        # input_size: int = 784,
        # lin1_size: int = 256,
        # lin2_size: int = 256,
        # lin3_size: int = 256,
        # output_size: int = 10,
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
        self.log(f"{mode}/ce", loss)
        self.train_acc.update(preds, y)
        self.log(f"{mode}/acc", self.train_acc.compute(), on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        mode = "val"
        loss, preds, y = self.step(batch)
        self.log(f"{mode}/ce", loss)
        self.val_acc.update(preds, y)
        self.log(f"{mode}/acc", self.val_acc.compute(), on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        mode = "test"
        loss, preds, y = self.step(batch)
        self.log(f"{mode}/ce", loss)
        self.val_acc.update(preds, y)
        self.log(f"{mode}/acc", self.val_acc.compute(), on_step=False, on_epoch=True)
        return loss

    def step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        # loss = self.criterion(logits, y)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

    def forward(self, x: torch.Tensor, k: int = 1):
        out = self.model.forward(x, k)  # B x k x ....
        if k == 1:
            return out.squeeze(1)
        return out
