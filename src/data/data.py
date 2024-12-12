# Adapterd from Pytorch Lighntning Bolts VisionDataModule  : https://github.com/PyTorchLightning/lightning-bolts
from typing import Generator, Optional, Sequence, Union

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST

from data.mio_dataset import MIOTCDDataset
from src.data.tox21_dataloader import get_tox21_dataloader  # Import Tox21 dataloader
from .active import ActiveLearningDataset
from .base_datamodule import BaseDataModule
from .longtail import create_imbalanced_dataset
from .skin_dataset import ISIC2016, ISIC2019
from .transformations import get_transform


class TorchVisionDM(BaseDataModule):
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
        persistent_workers: bool = True,
        imbalance: bool = False,
        timeout: int = 0,
        val_size: Optional[int] = None,
        balanced_sampling: bool = False,
    ):
        super().__init__(
            val_split=val_split,
            batch_size=batch_size,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
            min_train=min_train,
            active=active,
            random_split=random_split,
            seed=seed,
            persistent_workers=persistent_workers,
            timeout=timeout,
            val_size=val_size,
            balanced_sampling=balanced_sampling,
        )

        self.data_root = data_root
        self.dataset = dataset

        self.num_classes = num_classes

        self.mean = mean
        self.std = std
        self.shape = shape
        assert shape[-1] == len(mean)
        assert shape[-1] == len(std)

        ### Add Tox21 Support ###
        if self.dataset == "mnist":
            self.dataset_cls = MNIST
        elif self.dataset == "cifar10":
            self.dataset_cls = CIFAR10
        elif self.dataset == "cifar100":
            self.dataset_cls = CIFAR100
        elif self.dataset == "fashion_mnist":
            self.dataset_cls = FashionMNIST
        elif self.dataset == "isic2016":
            self.dataset_cls = ISIC2016
        elif self.dataset == "isic2019":
            self.dataset_cls = ISIC2019
        elif self.dataset == "miotcd":
            self.dataset_cls = MIOTCDDataset
        elif self.dataset == "tox21":  # Add Tox21 Dataset Case
            self.train_loader = get_tox21_dataloader(f"{self.data_root}/tox21_train_processed.pkl", batch_size=batch_size)
            self.val_loader = get_tox21_dataloader(f"{self.data_root}/tox21_test_processed.pkl", batch_size=batch_size)
        else:
            raise NotImplementedError

        # Call the setup for image-based datasets only
        if self.dataset not in ["tox21"]:
            self._setup_datasets()

        if not self.shuffle:
            raise ValueError("shuffle flag has to be set to true")

    def _setup_datasets(self):
        """Creates the active training dataset and validation and test datasets"""
        try:
            self.dataset_cls(root=self.data_root, download=False)
        except:  # Error is assumed here to stem from data not being present.
            # Download the TorchVision Dataset
            self.dataset_cls(root=self.data_root, download=True)
        self.train_set = self.dataset_cls(
            self.data_root, train=True, transform=self.train_transforms
        )
        self.train_set = self._split_dataset(self.train_set, train=True)

        if self.imbalance:
            self.train_set = create_imbalanced_dataset(
                self.train_set, imb_type="exp", imb_factor=0.02
            )

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

    def train_dataloader(self) -> DataLoader:
        if self.dataset == "tox21":
            return self.train_loader  # Return preloaded Tox21 train loader
        return self.get_dataloader(self.train_set, mode="train")

    def val_dataloader(self) -> DataLoader:
        if self.dataset == "tox21":
            return self.val_loader  # Return preloaded Tox21 validation loader
        return self.get_dataloader(self.val_set, mode="test")

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.test_set, mode="test")


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
