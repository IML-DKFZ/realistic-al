from copy import deepcopy
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .active import ActiveLearningDataset
from .base_datamodule import BaseDataModule
from .toy_data import *
from .transformations import get_transform


def make_toy_dataset(X: np.ndarray, y: np.ndarray):
    return TensorDataset(torch.from_numpy(X).to(dtype=torch.float), torch.from_numpy(y))


class ToyDataset(Dataset):
    def __init__(
        self, X: np.ndarray, y: np.ndarray, transform: Optional[Callable] = None
    ):
        """Dataset handling Toy Data with additional transforms.

        Args:
            X (np.ndarray): Predictors
            y (np.ndarray): Labeles
            transform (Optional[Callable], optional): Transform. Defaults to None.
        """
        self.data = torch.from_numpy(X).to(dtype=torch.float)
        self.targets = torch.from_numpy(y)
        self.transform = transform
        assert len(X) == len(y)

    def __getitem__(self, index):
        data = self.data[index]
        if self.transform is not None:
            data = self.transform(data)
        return (data, self.targets[index])

    def __len__(self):
        return len(self.data)


# TODO: check why persistent workers=True throws errors!
class ToyDM(BaseDataModule):
    def __init__(
        self,
        data_root: Union[str, None] = None,  # placeholder right now
        val_split: Union[float, int] = 0.2,
        num_samples: int = 1000,  # number of samples generated for pool training data and validation
        num_test_samples: int = 5000,  # number of samples generated for testing of model
        data_noise: float = 0.15,  # noise inherent for data generation
        batch_size: int = 64,
        dataset: str = "two_moons",
        drop_last: bool = False,
        num_workers: int = 12,
        pin_memory: bool = True,
        shuffle: bool = True,
        min_train: int = 5500,
        active: bool = True,
        random_split: bool = True,
        num_classes: int = 10,
        transform_test: str = "basic",
        transform_train: str = "basic",
        shape: Sequence = (2,),
        mean: Sequence = (0, 0),
        std: Sequence = (1, 1),
        seed: int = 12345,
        persistent_workers=False,
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
        )

        self.batch_size = batch_size
        self.dataset = dataset

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

        # Toy data specific variables
        self.data_noise = data_noise
        self.num_samples = num_samples
        self.num_test_samples = num_test_samples

        if self.dataset == "two_moons":
            self.dataset_generator = generate_moons_data
        elif self.dataset == "blob_4c":

            def gen_func(n_samples, noise=0.15, seed=12345):
                return generate_blob_data(n_samples, noise=noise, seed=seed, centers=4)

            self.dataset_generator = gen_func
        elif self.dataset == "blob_4":

            def gen_func(n_samples, noise=0.15, seed=12345):
                X, y = generate_blob_data(n_samples, noise=noise, seed=seed, centers=4)
                y = merge_labels(y, num_labels=2)
                return X, y

            self.dataset_generator = gen_func
        elif self.dataset == "circles":
            self.dataset_generator = generate_circles_data
        else:
            raise NotImplementedError
        self._setup_datasets()

        if not self.shuffle:
            raise ValueError("shuffle flag has to be set to true")

    def _setup_datasets(self):
        """Creates the active training dataset and validation and test datasets"""
        full_sample_count = self.num_samples + self.num_test_samples
        full_dataset = self.dataset_generator(
            n_samples=full_sample_count, noise=self.data_noise, seed=self.seed
        )

        train_data, test_data, train_label, test_label = train_test_split(
            *full_dataset, test_size=self.num_test_samples
        )

        # if:
        # basic version with train and validation split
        # self.train_set = make_toy_dataset(train_data, train_label)
        self.train_set = ToyDataset(
            train_data, train_label, transform=self.train_transforms
        )

        self.train_set = self._split_dataset(self.train_set, train=True)

        if self.active:
            self.train_set = ActiveLearningDataset(
                self.train_set, pool_specifics={"transform": self.test_transforms}
            )

        self.val_set = make_toy_dataset(train_data, train_label)
        # self.val_set = self._split_dataset(self.val_set, train=False)
        self.val_set = ToyDataset(
            train_data, train_label, transform=self.test_transforms
        )
        # self.test_set = make_toy_dataset(test_data, test_label)
        self.test_set = ToyDataset(
            test_data, test_label, transform=self.test_transforms
        )

        # else:
        # add cross validation version here!
        # pass

    def train_dataloader(self):
        return self.get_dataloader(self.train_set, mode="train")

    def val_dataloader(self):
        return self.get_dataloader(self.val_set, mode="test")

    def test_dataloader(self):
        return self.get_dataloader(self.val_set, mode="test")

    def create_dataloader(self, dataset, drop_last=False, shuffle=False):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
        )

    def labeled_dataloader(self, batch_size: Optional[int] = None):
        """Returns the dataloader for the labeled set with test time transformations"""
        if batch_size is None:
            batch_size = self.batch_size
        if hasattr(self.train_set, "labelled_set"):
            loader = DataLoader(
                self.train_set.labelled_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=self.drop_last,
            )
        else:
            labelled_dset = deepcopy(self.train_set)
            labelled_dset.transform = self.test_transforms
            loader = DataLoader(
                labelled_dset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=self.drop_last,
            )
        return loader


if __name__ == "__main__":
    from tqdm import tqdm

    for dataname, data in datasets.items():
        print(
            "Creating and Iterating over Dataloader for Data Type {}".format(dataname)
        )
        X, y = data
        try:
            toy_dataset = make_toy_dataset(X, y)
            for x, label in tqdm(toy_dataset):
                pass

            print("Succesful on Data Type {}".format(dataname))
        except Exception as e:
            print(e)
            print("Unsuccesful on Data Type {}".format(dataname))
