import pytorch_lightning as pl
from torch.functional import norm
from torch.utils import data
from models.fixmatch import FixMatch
from data.data import TorchVisionDM
import hydra
from omegaconf import DictConfig, open_dict
from utils import config_utils
from query import QuerySampler
import torch
import os
from typing import Union
import numpy as np
import gc

import utils

from run_training import TrainingLoop

active_dataset = True


@hydra.main(config_path="./config", config_name="config_fixmatch")
def main(cfg: DictConfig):
    config_utils.print_config(cfg)
    train(cfg)


def train(cfg: DictConfig):
    utils.set_seed(cfg.trainer.seed)
    balanced = cfg.active.balanced
    num_classes = cfg.data.num_classes
    num_labelled = cfg.active.num_labelled

    datamodule = TorchVisionDM(
        data_root=cfg.trainer.data_root,
        batch_size=cfg.trainer.batch_size,
        dataset=cfg.data.name,
        min_train=cfg.active.min_train,
        val_split=cfg.data.val_split,
        random_split=cfg.active.random_split,
        active=active_dataset,
        num_classes=cfg.data.num_classes,
        mean=cfg.data.mean,
        std=cfg.data.std,
        transform_train=cfg.data.transform_train,
        transform_test=cfg.data.transform_test,
        shape=cfg.data.shape,
    )
    # datamodule.prepare_data()
    # datamodule.setup()
    num_classes = cfg.data.num_classes
    if active_dataset:
        if balanced:
            datamodule.train_set.label_balanced(
                n_per_class=num_labelled // num_classes, num_classes=num_classes
            )
        else:
            datamodule.train_set.label_randomly(num_labelled)

    training_loop = TrainingLoop(cfg, datamodule, active=False)
    training_loop.main()

    # training_loop(cfg, datamodule, active=False)


class FixTrainingLoop(TrainingLoop):
    def init_model(self):
        self.model = FixMatch(self.cfg)


# def training_loop(
#     cfg: DictConfig,
#     datamodule: TorchVisionDM,
#     count: Union[None, int] = None,
#     active: bool = True,
# ):
#     utils.set_seed(cfg.trainer.seed)
#     train_iters_per_epoch = len(datamodule.train_set) // cfg.trainer.batch_size
#     with open_dict(cfg):
#         cfg.trainer.train_iters_per_epoch = train_iters_per_epoch

#     model = FixMatch(config=cfg)

#     if count is None:
#         version = cfg.trainer.experiment_id
#         name = cfg.trainer.experiment_name
#     else:
#         version = "loop-{}".format(count)
#         name = "{}/{}".format(cfg.trainer.experiment_name, cfg.trainer.experiment_id)
#     tb_logger = pl.loggers.TensorBoardLogger(
#         save_dir=cfg.trainer.experiments_root,
#         name=name,
#         version=version,
#     )

#     lr_monitor = pl.callbacks.LearningRateMonitor()
#     callbacks = [lr_monitor]
#     if datamodule.val_dataloader() is not None:
#         # ckpt_callback = pl.callbacks.ModelCheckpoint(monitor="val/loss", mode="min")
#         ckpt_callback = pl.callbacks.ModelCheckpoint(
#             monitor="val/acc",
#             mode="max",
#             save_last=True,
#         )
#     else:
#         ckpt_callback = pl.callbacks.ModelCheckpoint(monitor="train/acc", mode="max")
#     callbacks.append(ckpt_callback)
#     if cfg.trainer.early_stop:
#         early_stop_callback = pl.callbacks.EarlyStopping("val/acc", mode="max")
#         callbacks.append(early_stop_callback)

#     trainer = pl.Trainer(
#         gpus=cfg.trainer.n_gpus,
#         logger=tb_logger,
#         max_epochs=cfg.trainer.max_epochs,
#         min_epochs=cfg.trainer.min_epochs,
#         fast_dev_run=cfg.trainer.fast_dev_run,
#         terminate_on_nan=True,
#         callbacks=callbacks,
#         check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
#         progress_bar_refresh_rate=cfg.trainer.progress_bar_refresh_rate,
#         gradient_clip_val=cfg.trainer.gradient_clip_val,
#         precision=cfg.trainer.precision,
#         benchmark=cfg.trainer.deterministic is False,
#         deterministic=cfg.trainer.deterministic,
#     )
#     datamodule = model.wrap_dm(datamodule)
#     trainer.fit(model=model, datamodule=datamodule)
#     if not cfg.trainer.fast_dev_run:
#         best_path = ckpt_callback.best_model_path
#         print("Model for Testing is selected from path; {}".format(best_path))
#         model.load_from_checkpoint(best_path)

#         model = model.to("cuda:0")
#     test_results = trainer.test(model=model)
#     gc.collect()
#     torch.cuda.empty_cache()

#     if active:
#         query_sampler = QuerySampler(cfg, model, count=count)
#         query_sampler.setup()
#         stored = query_sampler.active_callback(datamodule)
#         return stored


if __name__ == "__main__":
    main()
