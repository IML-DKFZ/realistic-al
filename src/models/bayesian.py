from typing import Any, List, Optional
import math
from omegaconf import DictConfig

import torch
from models.networks import build_model
from .abstract_classifier import AbstractClassifier


class BayesianModule(AbstractClassifier):
    def __init__(self, config: DictConfig):
        """Simple Bayesian Neural Network which can be used with BatchBALD.
        For a Non-Bayesian Network set config.dopout_p to 0."""
        super().__init__(eman=True)
        self.save_hyperparameters(config)
        self.model = build_model(
            config, num_classes=config.data.num_classes, data_shape=config.data.shape
        )
        self.k = self.hparams.model.k
        if self.hparams.model.load_pretrained:
            self.load_from_ssl_checkpoint()
        self.init_ema_model(use_ema=config.model.use_ema)

    def training_step(self, batch, batch_idx):
        mode = "train"
        loss, preds, y = self.step(batch, k=1)
        self.log(f"{mode}/loss", loss)
        self.acc_train.update(preds, y)
        return loss
