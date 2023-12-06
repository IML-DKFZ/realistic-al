import gc
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
from loguru import logger
from omegaconf import DictConfig
from pytorch_lightning.callbacks import TQDMProgressBar

from data.data import TorchVisionDM
from models.bayesian import BayesianModule
from models.callbacks.metrics_callback import (
    ImbClassMetricCallback,
    ISIC2016MetricCallback,
)
from query import QuerySampler
from utils.io import save_json
from utils.log_utils import log_git


class ActiveTrainingLoop(object):
    def __init__(
        self,
        cfg: DictConfig,
        datamodule: TorchVisionDM,
        count: Union[None, int] = None,
        active: bool = True,
        base_dir: str = os.getcwd(),  # TODO: change this to some other value!
        loggers: str = True,
    ):
        # Class capturing the logic for Active Training Loops.
        self.cfg = cfg
        self.datamodule = deepcopy(datamodule)
        self.count = count
        self.active = active
        self.device = "cuda:0"
        self.base_dir = Path(base_dir)  # carries path to run
        self._save_dict = dict()
        self._init_model()
        self.ckpt_callback = self._init_ckpt_callback()
        self.callbacks = self._init_callbacks()
        self.loggers = False
        if loggers:
            self.loggers = self._init_loggers()

    def _init_ckpt_callback(self) -> pl.callbacks.ModelCheckpoint:
        ckpt_path = os.path.join(self.log_dir, "checkpoints")
        # TODO: Clean this up via selection via a config file!
        if self.datamodule.val_dataloader() is not None:
            if self.cfg.data.name == "isic2016":
                monitor = "val/auroc"
                mode = "max"
            elif "isic" in self.cfg.data.name:
                monitor = "val/w_acc"
                mode = "max"
            elif self.cfg.data.name == "miotcd":
                monitor = "val/w_acc"
                mode = "max"
            else:
                monitor = "val/acc"
                mode = "max"
            ckpt_callback = pl.callbacks.ModelCheckpoint(
                dirpath=ckpt_path,
                monitor=monitor,
                mode=mode,
                save_last=self.cfg.trainer.save_last,
            )
        else:
            ckpt_callback = pl.callbacks.ModelCheckpoint(
                dirpath=ckpt_path, monitor="train/acc", mode="max"
            )
        return ckpt_callback

    def _init_callbacks(self):
        lr_monitor = pl.callbacks.LearningRateMonitor()
        callbacks = [lr_monitor]
        callbacks.append(self.ckpt_callback)
        if self.cfg.trainer.early_stop and self.datamodule.val_dataloader is not None:
            callbacks.append(pl.callbacks.EarlyStopping("val/acc", mode="max"))
        if self.cfg.data.name == "isic2016":
            callbacks.append(ISIC2016MetricCallback())
        if self.cfg.data.name in ["isic2019", "miotcd"]:
            callbacks.append(
                ImbClassMetricCallback(num_classes=self.cfg.data.num_classes)
            )
        # add progress bar
        callbacks.append(
            TQDMProgressBar(refresh_rate=self.cfg.trainer.progress_bar_refresh_rate)
        )
        return callbacks

    @property
    def log_dir(self) -> Path:
        log_dir = Path(self.base_dir)
        if self.count is not None:
            log_dir = log_dir / self.version
        return log_dir

    @property
    def version(self) -> str:
        if self.count is not None:
            return "loop-{}".format(self.count)
        return self.cfg.trainer.experiment_id

    @property
    def name(self) -> str:
        name = self.cfg.trainer.experiment_name
        if self.count is not None:
            name = os.path.join(name, self.cfg.trainer.experiment_id)
        return name

    @property
    def data_ckpt_path(self) -> Path:
        return self.log_dir / "data_ckpt"

    @staticmethod
    def obtain_meta_data(repo_path: str, repo_name: str = "repo-name"):
        # based on: https://github.com/MIC-DKFZ/nnDetection/blob/6ac7dac6fd9ffd85b74682a2f565e0028305c2c0/scripts/train.py#L187-L226
        meta_data = {}
        meta_data["date"] = str(datetime.now())
        meta_data["git"] = log_git(repo_path)
        return meta_data

    def _init_loggers(self):
        tb_logger = pl.loggers.TensorBoardLogger(
            save_dir=self.cfg.trainer.experiments_root,
            name=self.name,
            version=self.version,
        )
        # add csv logger for important values!
        csv_logger = pl.loggers.CSVLogger(
            save_dir=self.cfg.trainer.experiments_root,
            name=self.name,
            version=self.version,
        )
        return [tb_logger, csv_logger]

    def _init_model(self):
        self.model = BayesianModule(self.cfg)

    def _init_trainer(self):
        self.trainer = pl.Trainer(
            gpus=self.cfg.trainer.n_gpus,
            logger=self.loggers,
            max_epochs=self.cfg.trainer.max_epochs,
            min_epochs=self.cfg.trainer.min_epochs,
            fast_dev_run=self.cfg.trainer.fast_dev_run,
            # detect_anomaly=True, # set this flag always to False for fast trainings!
            callbacks=self.callbacks,
            check_val_every_n_epoch=self.cfg.trainer.check_val_every_n_epoch,
            # progress_bar_refresh_rate=self.cfg.trainer.progress_bar_refresh_rate,
            gradient_clip_val=self.cfg.trainer.gradient_clip_val,
            precision=self.cfg.trainer.precision,
            benchmark=self.cfg.trainer.deterministic is False,
            deterministic=self.cfg.trainer.deterministic,
            profiler=self.cfg.trainer.profiler,
            # enable_progess_bar=self.cfg.trainer.enable_progress_bar,
        )

    def _fit(self):
        """Performs the fit, selects the best performing model and cleans up cache."""
        datamodule = self.model.wrap_dm(self.datamodule)
        self.model.setup_data_params(datamodule)
        self.trainer.fit(model=self.model, datamodule=datamodule)
        if not self.cfg.trainer.fast_dev_run and self.cfg.trainer.load_best_ckpt:
            best_path = self.ckpt_callback.best_model_path
            logger.info("Final Model from: {}".format(best_path))
            self.model = self.model.load_from_checkpoint(best_path)
        else:
            logger.info("Final Model from last iteration.")
        gc.collect()
        torch.cuda.empty_cache()

    def _test(self):
        self.trainer.test(model=self.model, datamodule=self.datamodule)

    def active_callback(self):
        """Execute active learning logic. -- not included in main.
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

    # TODO: Delete for open repostitory!
    # TODO: Saving in log_save_dict could be done via pickle allowing for more datatypes?
    def update_save_dict(self, sub_key: str, sub_dict: Dict[str, np.ndarray]):
        """Update the values of _save_dict with a new dictionary.

        Args:
            sub_key (str): Key which is added
            sub_dict (Dict[str, np.ndarray]): Dictionary which is added
        """
        for key, val in sub_dict.items():
            if not isinstance(val, np.ndarray):
                raise TypeError("sub_dict needs values of type np.ndarray")
            if not isinstance(key, str):
                raise TypeError("sub_dict needs keys of type str")
        self._save_dict[sub_key] = sub_dict

    # TODO: Delete for open repostitory!
    def log_save_dict(self):
        """Saves the values of _save_dict to log_dir"""
        for sub_key, sub_dict in self._save_dict.items():
            self.log_dict(sub_key, sub_dict, level="log", sub_folder="save_dict")

    # TODO: Delete for open repostitory!
    def log_dict(
        self,
        name: str,
        dictionary: Dict[str, np.ndarray],
        level: str = "log",
        sub_folder: Optional[str] = None,
        path: Optional[Union[str, Path]] = None,
        ending: str = "npz",
    ):
        """Save dictionary consisting of np.ndarrays according to log.
        Pattern: path/sub_folder/name.ending

        Args:
            name (str): basename of file
            dictionary (Dict[np.ndarray]): dictionary to save
            level (str, optional): "base" or "log". Defaults to "log".
            sub_folder (Optional[str], optional): Additional Path. Defaults to None.
            path (Optional[Union[str, Path]], optional): Path in which it is saved. Defaults to None.
            ending (str, optional): _description_. Defaults to "npz".

        Raises:
            ValueError: _description_
        """
        if path is None:
            if level == "base":
                path = self.base_dir
            elif level == "log":
                path = self.log_dir
            else:
                raise ValueError("level is neither base or log and path is None.")
        path = os.path.join(path, sub_folder)
        if os.path.isdir(path) is False:
            os.makedirs(path)
        save_file = os.path.join(path, "{}.{}".format(name, ending))
        np.savez_compressed(save_file, **dictionary)

    def _setup_log_struct(self):
        """Save Meta data to a json file."""
        meta_data = self.obtain_meta_data(
            os.path.dirname(os.path.abspath(__file__)), repo_name="realistic-al"
        )
        if os.path.exists(self.log_dir) is False:
            os.makedirs(self.log_dir)
        save_meta = self.log_dir / "meta.json"
        save_json(meta_data, save_meta)

    def main(self):
        """Executing logic of the Trainer.
        setup_..., init_..., fit, test, final_callback"""
        self._setup_log_struct()
        if self.active:
            self.datamodule.train_set.save_checkpoint(self.data_ckpt_path)
        self._init_trainer()
        self._fit()
        if self.trainer.interrupted:
            return
        if self.cfg.trainer.run_test:
            self._test()
        self.final_callback()
