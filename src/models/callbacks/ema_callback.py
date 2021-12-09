import pytorch_lightning as pl
from pytorch_lightning import Callback
import torch.nn as nn
from typing import Sequence


class EMAWeightUpdate(Callback):
    def __init__(self, tau: float = 0.999, eman: bool = False):
        super().__init__()
        self.tau = tau
        self.eman = eman

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        model = pl_module.model
        ema_model = pl_module.ema_model

        self.update_weights(model, ema_model)

    def update_weights(self, model: nn.Module, ema_model: nn.Module) -> None:
        for (name, model_p), (_, ema_p) in zip(
            model.named_parameters(), ema_model.named_parameters()
        ):
            ema_p.data = self.tau * ema_p.data + (1 - self.tau) * model_p.data

        for (name, model_b), (_, ema_b) in zip(
            model.named_buffers(), ema_model.named_buffers()
        ):
            if not self.eman:
                ema_b.data = model_b.data
            # the eman weight update for the batchnorm parameter leads to superior downstream
            # task performance according to https://arxiv.org/pdf/2101.08482.pdf
            if self.eman:
                ema_b.data = self.tau * ema_b.data + (1 - self.tau) * model_b.data
