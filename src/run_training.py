import pytorch_lightning as pl
from models.bayesian import BayesianModule
from data.data import TorchVisionDM
import hydra
from omegaconf import DictConfig, open_dict
from utils import config_utils
from query import QuerySampler
import torch
from typing import Union
import gc

import utils

active_dataset = True


@hydra.main(config_path="./config", config_name="config")
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
        num_workers=cfg.trainer.num_workers,
    )
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


class TrainingLoop(object):
    def __init__(
        self,
        cfg,
        datamodule: TorchVisionDM,
        count: Union[None, int] = None,
        active: bool = True,
    ):
        self.cfg = cfg
        self.datamodule = datamodule
        self.count = count
        self.active = active
        self.model = None
        self.logger = None
        self.ckpt_callback = None
        self.callbacks = None
        self.device = "cuda:0"

    def init_callbacks(self):
        lr_monitor = pl.callbacks.LearningRateMonitor()
        callbacks = [lr_monitor]
        if self.datamodule.val_dataloader() is not None:
            ckpt_callback = pl.callbacks.ModelCheckpoint(
                monitor="val/acc",
                mode="max",
                save_last=True,
            )
        else:
            ckpt_callback = pl.callbacks.ModelCheckpoint(
                monitor="train/acc", mode="max"
            )
        callbacks.append(ckpt_callback)
        if self.cfg.trainer.early_stop and self.datamodule.val_dataloader is not None:
            callbacks.append(pl.callbacks.EarlyStopping("val/acc", mode="max"))
        self.ckpt_callback = ckpt_callback
        self.callbacks = callbacks

    def init_logger(self):
        if self.count is None:
            version = self.cfg.trainer.experiment_id
            name = self.cfg.trainer.experiment_name
        else:
            version = "loop-{}".format(self.count)
            name = "{}/{}".format(
                self.cfg.trainer.experiment_name, self.cfg.trainer.experiment_id
            )
        tb_logger = pl.loggers.TensorBoardLogger(
            save_dir=self.cfg.trainer.experiments_root,
            name=name,
            version=version,
        )
        self.logger = tb_logger

    def init_model(self):
        self.model = BayesianModule(self.cfg)

    def init_trainer(self):
        self.trainer = pl.Trainer(
            gpus=self.cfg.trainer.n_gpus,
            logger=self.logger,
            max_epochs=self.cfg.trainer.max_epochs,
            min_epochs=self.cfg.trainer.min_epochs,
            fast_dev_run=self.cfg.trainer.fast_dev_run,
            terminate_on_nan=True,
            callbacks=self.callbacks,
            check_val_every_n_epoch=self.cfg.trainer.check_val_every_n_epoch,
            progress_bar_refresh_rate=self.cfg.trainer.progress_bar_refresh_rate,
            gradient_clip_val=self.cfg.trainer.gradient_clip_val,
            precision=self.cfg.trainer.precision,
            benchmark=self.cfg.trainer.deterministic is False,
            deterministic=self.cfg.trainer.deterministic,
            profiler=self.cfg.trainer.profiler,
            # enable_progess_bar=self.cfg.trainer.enable_progress_bar,
        )

    def fit(self):
        datamodule = self.model.wrap_dm(self.datamodule)
        self.trainer.fit(model=self.model, datamodule=datamodule)
        if not self.cfg.trainer.fast_dev_run:
            best_path = self.ckpt_callback.best_model_path
            print("Model for Testing is selected from path; {}".format(best_path))
            self.model.load_from_checkpoint(best_path)
        gc.collect()
        torch.cuda.empty_cache()

    def test(self):
        self.trainer.test(model=self.model, datamodule=self.datamodule)

    def active_callback(self):
        self.model = self.model.to(self.device)
        query_sampler = QuerySampler(self.cfg, self.model, count=self.count)
        query_sampler.setup()
        stored = query_sampler.active_callback(self.datamodule)
        return stored

    def main(self):
        self.init_model()
        self.init_callbacks()
        self.init_logger()
        self.init_trainer()
        self.fit()
        self.test()


if __name__ == "__main__":
    main()
