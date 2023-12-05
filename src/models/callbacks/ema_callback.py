from typing import Sequence

import pytorch_lightning as pl
import torch.nn as nn
from pytorch_lightning import Callback


class EMAWeightUpdate(Callback):
    def __init__(self, tau: float = 0.999, eman: bool = False):
        """Callback which updates the ema model in a pl.Module.
        EMA is implemented according to MOCO and BYOL by Pytorch Lightning Bolts.

        if eman is true it also updates the BN parameters according to:
        EMAN: Exponential Moving Average Normalization for Self-supervised and Semi-supervised Learning
        ArXiv: https://arxiv.org/abs/2101.08482
        Repo: https://github.com/amazon-research/exponential-moving-average-normalization

        Args:
            tau (float, optional): ema rate. Defaults to 0.999.
            eman (bool, optional): use EMAN instead of EMA. Defaults to False.
        """
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
