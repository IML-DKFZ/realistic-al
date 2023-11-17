from typing import Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import (
    DataLoader,
    Dataset,
    Subset,
    WeightedRandomSampler,
    random_split,
)

from data.active import ActiveLearningDataset
from data.utils import (
    ActiveSubset,
    RandomFixedLengthSampler,
    activesubset_from_subset,
    seed_worker,
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
        persistent_workers: bool = True,
        timeout: int = 0,
        val_size: Optional[int] = None,
        balanced_sampling: bool = False,
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
        self.val_size = val_size
        self.balanced_sampling = balanced_sampling

        # Used for the traning validation split
        self.seed = seed
        self.indices = None

        if not self.shuffle:
            raise ValueError("shuffle flag has to be set to true")

        self.timeout = timeout

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
            if self.val_size:
                if self.dataset == "isic2019":
                    # Draw the first 200 validation samples in a balanced fashion
                    val_size = 200
                elif self.dataset == "miotcd":
                    # Draw the first 25xnum_classes validation samples in a balanced fashion
                    val_size = 11 * 25
                else:
                    val_size = self.val_size
                indices = []
                labels = []
                # for (x, y) in dataset_val:
                #     labels.append(y)
                # labels = np.array(labels)
                labels = dataset_val.targets
                assert (
                    len(labels.shape) == 1
                )  # This does currently only work for single labels

                n_per_class = val_size / len(np.unique(labels))
                assert n_per_class % 1 == 0
                n_per_class = int(n_per_class)
                for c in np.unique(labels):
                    class_indices = np.where(labels == c)[0]
                    indices.append(class_indices[:n_per_class])
                indices = np.concatenate(indices, axis=0)
                if self.dataset in ["isic2019", "miotcd"]:
                    # Draw all subsequent validation samples in a random fashion
                    val_size = self.val_size - val_size
                    if val_size > 0:
                        indices_new = np.arange(len(labels))
                        # no overlap between the balanced and random samples.
                        indices_new = indices_new[~np.isin(indices_new, indices)]
                        indices_new = np.random.choice(
                            indices_new, size=val_size, replace=False
                        )
                        indices = np.concatenate([indices, indices_new])
                dataset_val = ActiveSubset(dataset_val, indices)
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
                if self.balanced_sampling:
                    if isinstance(dataset, ActiveLearningDataset):
                        targets = dataset.labelled_set.targets
                    elif hasattr(dataset, "targets"):
                        targets = dataset.targets
                    else:
                        raise ValueError(
                            "dataset is neither an Active Learning dataset or has attribute targets"
                        )

                    classes, counts = np.unique(targets, return_counts=True)
                    weights = np.ones(len(dataset))
                    for cls, count in zip(classes, counts):
                        weights[targets == cls] = 1 / (count * len(classes))

                    return DataLoader(
                        dataset,
                        batch_size=self.batch_size,
                        sampler=WeightedRandomSampler(
                            weights, num_samples=self.min_train
                        ),
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
                        sampler=RandomFixedLengthSampler(dataset, self.min_train),
                        num_workers=self.num_workers,
                        pin_memory=self.pin_memory,
                        drop_last=self.drop_last,
                        worker_init_fn=seed_worker,
                        persistent_workers=self.persistent_workers,
                    )
            else:
                if self.balanced_sampling:
                    if isinstance(dataset, ActiveLearningDataset):
                        targets = dataset.labelled_set.targets
                    elif hasattr(dataset, "targets"):
                        targets = dataset.targets
                    else:
                        raise ValueError(
                            "dataset is neither an Active Learning dataset or has attribute targets"
                        )

                    classes, counts = np.unique(targets, return_counts=True)
                    weights = np.ones(len(dataset))
                    for cls, count in zip(classes, counts):
                        #
                        weights[targets == cls] = 1 / (count * len(classes))

                    return DataLoader(
                        dataset,
                        batch_size=self.batch_size,
                        sampler=WeightedRandomSampler(
                            weights, num_samples=len(dataset)
                        ),
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
        given size of the dataset. For labeling the pool - get indices with get_pool_indices
        """
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
