from typing import Optional, Sequence, Union

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset, random_split

from .active import ActiveLearningDataset

# from .data import TorchVisionDM
from .random_fixed_length_sampler import RandomFixedLengthSampler
from .toy_data import *
from .transformations import get_transform
from .utils import ActiveSubset, activesubset_from_subset, seed_worker
from torch.utils.data import Dataset


def make_toy_dataset(X: np.ndarray, y: np.ndarray):
    return TensorDataset(torch.from_numpy(X).to(dtype=torch.float), torch.from_numpy(y))


# this might also be implemented simply carrying
class ToyDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, transform=None):
        self.predictors = torch.from_numpy(X).to(dtype=torch.float)
        self.labels = torch.from_numpy(y)
        self.transform = transform
        assert len(X) == len(y)

    def __getitem__(self, index):
        data = self.predictors[index]
        if self.transform is not None:
            data = self.transform(data)
        return (data, self.labels[index])

    def __len__(self):
        return len(self.predictors)


class ToyDM(pl.LightningDataModule):
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

        # Toy data specific variables
        self.data_noise = data_noise
        self.num_samples = num_samples
        self.num_test_samples = num_test_samples

        if self.dataset == "two_moons":
            self.dataset_generator = generate_moons_data
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

        if True:
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
        else:
            # add cross validation version here!
            pass

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

    def create_dataloader(self, dataset, drop_last=False, shuffle=False):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def pool_dataloader(self, batch_size=64, m: Optional[int] = None):
        """Returns the dataloader for the pool with test time transformations and optional the
        given size of the dataset. For labeling the pool - get indices with get_pool_indices"""
        if hasattr(self.train_set, "pool"):
            pool = self.train_set.pool
        else:
            raise TypeError(
                "Training Set does not have the attribute pool. \n Try to use the ActiveDataset (by enabling active)"
            )
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
