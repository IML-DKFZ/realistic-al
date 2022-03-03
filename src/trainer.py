import os

import pytorch_lightning as pl
from models.bayesian import BayesianModule
from data.data import TorchVisionDM
from query import QuerySampler
import torch
from typing import Union
import gc
from datetime import datetime
from utils.log_utils import log_git
from utils.io import save_json


class ActiveTrainingLoop(object):
    def __init__(
        self,
        cfg,
        datamodule: TorchVisionDM,
        count: Union[None, int] = None,
        active: bool = True,
        base_dir: str = os.getcwd(),
    ):
        """Class capturing the logic for Active Training Loops."""
        self.cfg = cfg
        self.datamodule = datamodule
        self.count = count
        self.active = active
        self.model = None
        self.logger = None
        self.ckpt_callback = None
        self.callbacks = None
        self.device = "cuda:0"
        self.base_dir = base_dir
        self.log_dir = None
        self.init_paths()

    def init_callbacks(self):
        """Initializing the Callbacks used in Pytorch Lightning."""
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

    def init_paths(self):
        self.version = self.cfg.trainer.experiment_id
        self.name = self.cfg.trainer.experiment_name
        self.log_dir = self.base_dir
        if self.count is not None:
            self.version = "loop-{}".format(self.count)
            self.name = os.path.join(
                self.cfg.trainer.experiment_name, self.cfg.trainer.experiment_id
            )
            # here might appear errors for active learning
            self.log_dir = os.path.join(self.base_dir, self.name)

    @staticmethod
    def obtain_meta_data(repo_path: str, repo_name: str = "repo-name"):
        # based on: https://github.com/MIC-DKFZ/nnDetection/blob/6ac7dac6fd9ffd85b74682a2f565e0028305c2c0/scripts/train.py#L187-L226
        meta_data = {}
        meta_data["date"] = str(datetime.now())
        meta_data["git"] = log_git(repo_path, repo_name=repo_name)
        return meta_data

    def init_logger(self):
        """Initialize the Loggers used in Pytorch Lightning.
        Loggers initialized: Tensorobard Loggers
        Name scheme for logs:
        - if no count - root/name/id
        - if count - root/name/id/loop-{count}

        Note: hydra always logs in id folder"""
        tb_logger = pl.loggers.TensorBoardLogger(
            save_dir=self.cfg.trainer.experiments_root,
            name=self.name,
            version=self.version,
        )
        # add csv logger for important values!
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
            print("Model for Testing is selected from path: {}".format(best_path))
            self.model.load_from_checkpoint(best_path)
        gc.collect()
        torch.cuda.empty_cache()

    def test(self):
        self.trainer.test(model=self.model, datamodule=self.datamodule)

    def active_callback(self):
        """Execute active learning logic.
        Returns the queries to the oracle."""
        self.model = self.model.to(self.device)
        query_sampler = QuerySampler(
            self.cfg, self.model, count=self.count, device=self.device
        )
        query_sampler.setup()
        stored = query_sampler.active_callback(self.datamodule)
        return stored

    def final_callback(self):
        pass

    def setup_log_struct(self):
        meta_data = self.obtain_meta_data(
            os.path.dirname(os.path.abspath(__file__)), repo_name="active-playground"
        )
        if os.path.exists(self.log_dir) is False:
            os.makedirs(self.log_dir)
        save_meta = os.path.join(self.log_dir, "meta.json")
        save_json(meta_data, save_meta)

    def main(self):
        """Executing logic of the Trainer"""
        self.setup_log_struct()
        self.init_logger()
        self.init_model()
        self.init_callbacks()
        self.init_trainer()
        self.fit()
        self.test()
        self.final_callback()
        # add a wrap up!

        os.chdir(self.base_dir)
