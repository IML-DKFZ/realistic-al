# Adapterd from Pytorch Lighntning Bolts VisionDataModule  : https://github.com/PyTorchLightning/lightning-bolts
from typing import Generator, Optional, Sequence, Union

import numpy as np
import pytorch_lightning as pl
import torch
from .random_fixed_length_sampler import RandomFixedLengthSampler
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST

from .active import ActiveLearningDataset
from .utils import activesubset_from_subset, ActiveSubset, seed_worker
from .transformations import get_transform


class TorchVisionDM(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        val_split: Union[float, int] = 0.2,
        batch_size: int = 64,
        dataset: str = "mnist",
        drop_last: bool = False,
        num_workers: int = 12,
        pin_memory: bool = True,
        shuffle: bool = True,
        min_train: int = 5500,
        active: bool = True,
        random_split: bool = True,
        num_classes: int = 10,
        transform_train: str = "basic",
        transform_test: str = "basic",
        shape: Sequence = [28, 28, 1],
        mean: Sequence = (0,),
        std: Sequence = (1,),
        seed: int = 12345,
    ):
        super().__init__()

        self.data_root = data_root
        self.batch_size = batch_size
        self.dataset = dataset
        self.val_split = val_split

        self.drop_last = drop_last
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.min_train = min_train
        self.active = active
        self.random_split = random_split
        self.num_classes = num_classes

        # Used for the traning validation split
        self.seed = seed

        # TODO tidy up and generalize selection of transformations for more datasets
        self.mean = mean
        self.std = std
        self.shape = shape
        assert shape[-1] == len(mean)
        assert shape[-1] == len(std)

        self.train_transforms = get_transform(
            transform_train, self.mean, self.std, self.shape
        )
        self.test_transforms = get_transform(
            transform_test, self.mean, self.std, self.shape
        )

        if self.dataset == "mnist":
            self.dataset_cls = MNIST
        elif self.dataset == "cifar10":
            self.dataset_cls = CIFAR10
        elif self.dataset == "cifar100":
            self.dataset_cls = CIFAR100
        elif self.dataset == "fashion_mnist":
            self.dataset_cls = FashionMNIST
        else:
            raise NotImplementedError
        self._setup_datasets()

        if not self.shuffle:
            raise ValueError("shuffle flag has to be set to true")

    def _setup_datasets(self):
        """Creates the active training dataset and validation and test datasets"""
        try:
            self.dataset_cls(root=self.data_root, download=False)
        except:  # TODO: add error case for data not found here
            """Download the TorchVision Dataset"""
            self.dataset_cls(root=self.data_root, download=True)
        self.train_set = self.dataset_cls(
            self.data_root, train=True, transform=self.train_transforms
        )
        self.train_set = self._split_dataset(self.train_set, train=True)

        if self.active:
            self.train_set = ActiveLearningDataset(
                self.train_set, pool_specifics={"transform": self.test_transforms}
            )

        self.val_set = self.dataset_cls(
            self.data_root, train=True, transform=self.test_transforms
        )
        self.val_set = self._split_dataset(self.val_set, train=False)
        self.test_set = self.dataset_cls(
            self.data_root, train=False, transform=self.test_transforms
        )

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

    def train_dataloader(self):
        if len(self.train_set) <= self.min_train:
            return DataLoader(
                self.train_set,
                batch_size=self.batch_size,
                sampler=RandomFixedLengthSampler(self.train_set, self.min_train),
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=self.drop_last,
                worker_init_fn=seed_worker,
            )
        else:
            return DataLoader(
                self.train_set,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=self.drop_last,
                worker_init_fn=seed_worker,
            )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def pool_dataloader(self, batch_size=64, m: Optional[int] = None):
        """Returns the dataloader for the pool with test time transformations and optional the
        given size of the dataset. For labeling the pool - get indices with get_pool_indices"""
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


if __name__ == "__main__":
    import os

    data_root = os.getenv("DATA_ROOT")
    dm = TorchVisionDM(data_root=data_root, dataset="cifar10")
    dm.prepare_data()
    dm.setup()
    dm.train_set.label_randomly(6000)
    train_loader = dm.train_dataloader()
    labeled_loader = dm.labeled_dataloader()
    pool_loader = dm.pool_dataloader()

    # import matplotlib.pyplot as plt
    # import torchvision

    # def vis_first_batch(loader):
    #     x = next(iter(loader))[0]
    #     x = x - x.min()
    #     x = x / x.max()
    #     vis = torchvision.utils.make_grid(x).permute(1, 2, 0)
    #     plt.imshow(vis)
    #     plt.show()

    # import IPython

    # IPython.embed()
    from tqdm.auto import tqdm

    # print("Iterating over Labelled Training Set")
    # for x in tqdm(train_loader):
    #     pass
    # print("Iterating over Unlabelled Pool of Data")
    # for y in tqdm(DataLoader(dm.train_set.pool, batch_size=64)):
    #     pass
    # print("Iterating over the Validation Dataset")
    # val_loader = dm.val_dataloader()
    # for x in tqdm(val_loader):
    #     pass

    pool_loader = dm.pool_dataloader(m=20)
    inds = np.array([i for i in range(10)])
    pool_indices = dm.get_pool_indices(inds)
    assert np.array_equal(pool_indices, dm.indices[: len(pool_indices)])

    print(pool_indices)
