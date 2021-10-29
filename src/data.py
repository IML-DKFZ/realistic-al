# Inspired by Pytorch Lighntning Bolts VisionDataModule  : https://github.com/PyTorchLightning/lightning-bolts

import torch
from typing import Union

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST
from active import ActiveLearningDataset
from random_fixed_length_sampler import RandomFixedLengthSampler

SEED = 12345


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

        # Used for the traning validation split
        self.seed = SEED

        # TODO tidy up and generalize selection of transformations for more datasets
        if self.dataset in ["mnist", "fashion_mnist"]:
            self.train_transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            )
            self.test_transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            )

        elif self.dataset in ["cifar10", "cifar100"]:
            self.train_transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.RandomCrop(32, 4),
                    transforms.RandomHorizontalFlip(),
                    transforms.Normalize(
                        [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
                    ),
                ]
            )
            self.test_transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
                    ),
                ]
            )

        if self.dataset == "mnist":
            self.dataset_cls = MNIST
        elif self.dataset == "cifar10":
            self.dataset_cls = CIFAR10
        elif self.dataset == "cifar100":
            self.dataset_cls = CIFAR100
        elif self.dataest == "fashion_mnist":
            self.dataset_cls = FashionMNIST
        else:
            raise NotImplementedError

        if not self.shuffle:
            raise ValueError("shuffle flag has to be set to true")

    def prepare_data(self):
        """Download the TorchVision Dataset"""
        self.dataset_cls(root=self.data_root, download=True)

    def setup(self, stage="train"):
        """Creates the active training dataset and validation and test datasets"""
        self.train_set = self.dataset_cls(
            self.data_root, train=True, transform=self.train_transforms
        )
        self.train_set = self._split_dataset(self.train_set, train=True)

        # TODO think about better position to add this into the data-module
        if self.active:
            self.train_set = ActiveLearningDataset(
                self.train_set,
                #  TODO: check how to change the transform of a submodule!
                # pool_specifics={
                #     'transform' : self.test_transforms
                # }
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
        else:
            dataset_train = torch.utils.data.Subset(dataset,range(splits[0]))
            dataset_val = torch.utils.data.Subset(dataset, range(splits[0], splits[1]))
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
                # shuffle=self.shuffle,
                sampler=RandomFixedLengthSampler(self.train_set, self.min_train),
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=self.drop_last,
            )
        else:
            return DataLoader(
                self.train_set,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                # sampler=RandomFixedLengthSampler(self.train_set, self.min_train),
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=self.drop_last,
            )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

    def pool_dataloader(self, batch_size=64):
        return DataLoader(
            self.train_set.pool,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )


# TODO: make this function usable for these dataset
def ActiveSubset(Subset):
    """Subclass of torch Subset with direct access to transforms the underlying Dataset"""

    @property
    def transform(self):
        return self.dataset.transform

    @transform.setter
    def transform(self, new_transform):
        self.dataset.transform = new_transform


if __name__ == "__main__":
    import os

    data_root = os.getenv("DATA_ROOT")
    dm = TorchVisionDM(data_root=data_root)
    dm.prepare_data()
    dm.setup()
    dm.train_set.label_randomly(100)
    train_loader = dm.train_dataloader()
    from tqdm.auto import tqdm

    print("Iterating over Labelled Training Set")
    for x in tqdm(train_loader):
        pass
    print("Iterating over Unlabelled Pool of Data")
    for y in tqdm(DataLoader(dm.train_set.pool, batch_size=64)):
        pass

    print("Iterating over the Validation Dataset")
    val_loader = dm.val_dataloader()
    for x in tqdm(val_loader):
        pass
