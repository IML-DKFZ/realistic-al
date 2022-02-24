import pytorch_lightning as pl
from models.bayesian import BayesianModule
from data.data import TorchVisionDM
from query import QuerySampler
import torch
from typing import Union
import gc


class ActiveTrainingLoop(object):
    def __init__(
        self,
        cfg,
        datamodule: TorchVisionDM,
        count: Union[None, int] = None,
        active: bool = True,
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

    def init_logger(self):
        """Initialize the Loggers used in Pytorch Lightning.
        Loggers initialized: Tensorobard Loggers
        Name scheme for logs:
        - if no count - root/name/id
        - if count - root/name/id/count

        Note: hydra always logs in id folder"""
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
            print("Model for Testing is selected from path: {}".format(best_path))
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
