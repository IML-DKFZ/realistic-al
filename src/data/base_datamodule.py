from typing import Union, Optional
from abc import abstractmethod

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import pytorch_lightning as pl

from data.active import ActiveLearningDataset
from data.utils import (
    ActiveSubset,
    seed_worker,
    RandomFixedLengthSampler,
    activesubset_from_subset,
)


class BaseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        val_split: Union[float, int] = 0.2,
        batch_size: int = 64,
        drop_last: bool = False,
        num_workers: int = 12,
        pin_memory: bool = True,
        shuffle: bool = True,
        min_train: int = 5500,
        active: bool = True,
        random_split: bool = True,
        seed: int = 12345,
        persistent_workers=True,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.val_split = val_split

        self.drop_last = drop_last
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.min_train = min_train
        self.active = active
        self.random_split = random_split
        self.persistent_workers = persistent_workers

        # Used for the traning validation split
        self.seed = seed
        self.indices = None

        if not self.shuffle:
            raise ValueError("shuffle flag has to be set to true")

    # @property
    # @abstractmethod
    # def train_set(self) -> Dataset:
    #     pass

    # @property
    # @abstractmethod
    # def val_set(self) -> Dataset:
    #     pass

    # @property
    # def test_set(self) -> Dataset:
    #     raise NotImplementedError("There is currently no Test Set defined")

    def _split_dataset(self, dataset: Dataset, train: bool = True):
        """Splits the dataset into train and validation set."""
        len_dataset = len(dataset)  # type: ignore[arg-type]
        splits = self._get_splits(len_dataset)
        if self.random_split:
            dataset_train, dataset_val = random_split(
                dataset, splits, generator=torch.Generator().manual_seed(self.seed)
            )
            dataset_train = activesubset_from_subset(dataset_train)
            dataset_val = activesubset_from_subset(dataset_val)
        else:
            dataset_train = ActiveSubset(dataset, range(splits[0]))
            dataset_val = ActiveSubset(dataset, range(splits[0], splits[1]))
        if train:
            return dataset_train
        return dataset_val

    def _get_splits(self, len_dataset):
        """Computes split lengths for train and validation set."""
        if isinstance(self.val_split, int):
            train_len = len_dataset - self.val_split
            splits = [train_len, self.val_split]
        elif isinstance(self.val_split, float):
            val_len = int(self.val_split * len_dataset)
            train_len = len_dataset - val_len
            splits = [train_len, val_len]
        else:
            raise ValueError(f"Unsupported type {type(self.val_split)}")

        return splits

    def get_dataloader(self, dataset: Dataset, mode="train"):
        if mode == "train":
            if len(dataset) < self.min_train:
                return DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    sampler=RandomFixedLengthSampler(dataset, self.min_train),
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    drop_last=self.drop_last,
                    worker_init_fn=seed_worker,
                    persistent_workers=self.persistent_workers,
                )
            else:
                return DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=self.shuffle,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    drop_last=self.drop_last,
                    worker_init_fn=seed_worker,
                    persistent_workers=self.persistent_workers,
                )
        elif mode == "test":
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=False,
                persistent_workers=self.persistent_workers,
            )

    def pool_dataloader(self, batch_size=64, m: Optional[int] = None):
        """Returns the dataloader for the pool with test time transformations and optional the
        given size of the dataset. For labeling the pool - get indices with get_pool_indices"""
        if not self.active:
            raise TypeError(
                "Training Set does not have the attribute pool. \n Try to use the ActiveDataset (by enabling active)"
            )
        pool = self.train_set.pool
        self.indices = np.arange(len(pool), dtype=np.int)

        if m:
            if m > 0:
                m = min(len(pool), m)
                self.indices = np.random.choice(
                    np.arange(len(pool), dtype=np.int), size=m, replace=False
                )
                return DataLoader(
                    Subset(pool, indices=self.indices),
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    drop_last=self.drop_last,
                )

        return DataLoader(
            pool,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
        )

    def get_pool_indices(self, inds: np.ndarray):
        """Returns the indices of the underlying pool given the indices from the pool loader"""
        if not self.active:
            raise TypeError(
                "Training Set does not have the attribute pool. \n Try to use the ActiveDataset (by enabling active)"
            )
        if self.indices is None:
            raise ValueError("Currently Indices have not been set.")
        return self.indices[inds]

    def labeled_dataloader(self, batch_size=64):
        """Returns the dataloader for the labeled set with test time transformations"""
        loader = DataLoader(
            self.train_set.labelled_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )
        return loader
