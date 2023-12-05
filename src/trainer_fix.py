import gc
from typing import Optional

import pytorch_lightning as pl
import torch
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from loguru import logger
from torch.utils.data import DataLoader

from data.utils import ConcatDataloader
from models.fixmatch import FixMatch
from trainer import ActiveTrainingLoop


class ConcatenatedAugmenters:
    def __init__(self, augmenters):
        """Allows to use BatchGenerators Augmenters as a drop-in replacement for DataLoaders in
        Pytorch-Lightning.

        Args:
            augmenters (MultiThreadedAugmenter): _description_
        """
        self.augmenters = augmenters

    def __iter__(self):
        return self

    def __next__(self):
        batch = []
        for i, dl in enumerate(self.augmenters):
            data = next(dl)
            batch.append(data)
        return batch


class DataLoaderWrapper(SlimDataLoaderBase):
    def __init__(
        self,
        data_loader: DataLoader,
        batch_size: int,
        number_of_threads_in_multithreaded: Optional[int] = None,
    ):
        """Wrapper of Torch DataLoaders to be used in BatchGenerators with the MultiThreadedAugmenter.
        Iterates over the data_loader infinetely often.

        Args:
            data_loader (_type_): _description_
            batch_size (int): _description_
            number_of_threads_in_multithreaded (Optional[int], optional): _description_. Defaults to None.
        """
        super().__init__(
            iter(data_loader), batch_size, number_of_threads_in_multithreaded
        )
        self.data_loader = data_loader

    def __next__(self):
        try:
            return self._data.next()
        except:
            self._data = iter(self.data_loader)
            return self._data.next()


class FixTrainingLoop(ActiveTrainingLoop):
    def _init_model(self):
        self.model = FixMatch(self.cfg)

    def _init_trainer(self):
        self.trainer = pl.Trainer(
            gpus=self.cfg.trainer.n_gpus,
            logger=self.loggers,
            max_epochs=self.cfg.trainer.max_epochs,
            min_epochs=self.cfg.trainer.min_epochs,
            fast_dev_run=self.cfg.trainer.fast_dev_run,
            # detect_anomaly=True,
            callbacks=self.callbacks,
            check_val_every_n_epoch=self.cfg.trainer.check_val_every_n_epoch,
            gradient_clip_val=self.cfg.trainer.gradient_clip_val,
            precision=self.cfg.trainer.precision,
            benchmark=self.cfg.trainer.deterministic is False,
            deterministic=self.cfg.trainer.deterministic,
            profiler=self.cfg.trainer.profiler,
            limit_train_batches=self.cfg.trainer.train_iters_per_epoch,
        )

    def _fit(self):
        """Performs the fit, selects the best performing model and cleans up cache."""
        num_workers = self.datamodule.num_workers
        pin_memory = self.datamodule.pin_memory
        persistent_workers = self.datamodule.persistent_workers
        timeout = self.datamodule.timeout

        # if self.datamodule.dataset in ["miotcd", "isic2019"]:
        if True:
            logger.info("Use Pytorch DataLoader multiprocessing")
            datamodule = self.model.wrap_dm(self.datamodule)
            loader_label, loader_pool = datamodule.train_dataloader()
            train_dataloader = ConcatDataloader(loader_label, loader_pool)
        else:
            logger.info("Use BatchGenerators Augmenters multiprocessing")
            self.datamodule.persistent_workers = False
            self.datamodule.num_workers = 0
            self.datamodule.pin_memory = False
            self.datamodule.timeout = 0

            datamodule = self.model.wrap_dm(self.datamodule)
            loader_label, loader_pool = datamodule.train_dataloader()
            worker_label = num_workers // 2
            worker_pool = num_workers // 2

            multi_label = MultiThreadedAugmenter(
                DataLoaderWrapper(loader_label, None, worker_label),
                None,
                worker_label,
                pin_memory=False,
                timeout=timeout,
            )
            multi_pool = MultiThreadedAugmenter(
                DataLoaderWrapper(loader_pool, None, worker_pool),
                None,
                worker_pool,
                pin_memory=False,
                timeout=timeout,
            )

            self.datamodule.num_workers = num_workers
            self.datamodule.pin_memory = pin_memory
            self.datamodule.persistent_workers = persistent_workers
            self.datamodule.timeout = timeout

            train_dataloader = ConcatenatedAugmenters([multi_label, multi_pool])

        val_dataloader = datamodule.val_dataloader()

        self.model.setup_data_params(self.datamodule)
        self.model.train_iters_per_epoch = self.cfg.trainer.train_iters_per_epoch
        self.trainer.fit(
            model=self.model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
        if not self.cfg.trainer.fast_dev_run and self.cfg.trainer.load_best_ckpt:
            best_path = self.ckpt_callback.best_model_path
            logger.info("Final Model from: {}".format(best_path))
            self.model = self.model.load_from_checkpoint(best_path)
        else:
            logger.info("Final Model from last iteration.")
        gc.collect()
        torch.cuda.empty_cache()
