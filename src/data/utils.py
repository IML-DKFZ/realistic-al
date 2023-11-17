import random
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, Subset
from torchvision import transforms

from utils.tensor import to_numpy

from .randaugment import RandAugmentMC, RandAugmentMCCutout


class RandomFixedLengthSampler(Sampler):
    """
    Sometimes, you really want to do more with little data without increasing the number of epochs.

    This sampler takes a `dataset` and draws `target_length` samples from it (with repetition).
    From: https://github.com/BlackHC/BatchBALD/blob/master/src/random_fixed_length_sampler.py
    """

    def __init__(self, dataset: Dataset, target_length):
        super().__init__(dataset)
        self.dataset = dataset
        self.target_length = target_length

    def __iter__(self):
        # Ensure that we don't lose data by accident.
        if self.target_length < len(self.dataset):
            return iter(torch.randperm(len(self.dataset)))

        return iter((torch.randperm(self.target_length) % len(self.dataset)).tolist())

    def __len__(self):
        return self.target_length


class ActiveSubset(Subset):
    """Subclass of torch Subset with direct access to transforms the underlying Dataset and the targets."""

    @property
    def transform(self):
        return self.dataset.transform

    @transform.setter
    def transform(self, new_transform):
        self.dataset.transform = new_transform

    @property
    def targets(self) -> np.ndarray:
        targets = to_numpy(self.dataset.targets)
        return targets[self.indices]


def activesubset_from_subset(subset: Subset) -> ActiveSubset:
    return ActiveSubset(dataset=subset.dataset, indices=subset.indices)


class ConcatDataloader:
    """
    ConcatDataloader is an iterator that concatenates several data loaders.
    In each iteration, a batch from each data loader is drawn and added to the list of the concatenated batch.
    The iterator expires after the longest data loader expires.
    By: Maximillian Zenk
    """

    def __init__(self, *data_loaders):
        self.data_loaders = data_loaders
        self.iterators = []

    # def __len__(self):
    #     # other data loaders are repeated to match the longest one
    #     return max([len(d) for d in self.data_loaders])

    def __iter__(self):
        self.iterators = [iter(dl) for dl in self.data_loaders]
        return self

    def __next__(self):
        batch = []
        for i, dl in enumerate(self.iterators):
            data = []
            # could happen that empty data loader is passed
            if len(self.data_loaders[i]) > 0:
                try:
                    data = next(dl)
                except StopIteration:
                    # if the first dataloader in self.dataloaders is not the longest
                    # this could lead to an unnecessary iterator creation.
                    # However, this seems not to be a problem right now.
                    # if len(dl) == len(self):
                    #     # the longest data iterator has reached the end
                    #     raise StopIteration
                    # re-initialize data loader iterators if necessary
                    dl = iter(self.data_loaders[i])
                    self.iterators[i] = dl
                    data = next(dl)
            batch.append(data)

        if len(batch) == 1:
            batch = batch[0]
        return batch


class MultiHeadedTransform(object):
    def __init__(self, transforms: Iterable):
        """Multi Headed Transform on Data.

        Args:
            transforms (Iterable): Data Transformations
        """
        self.transforms = transforms

    def __call__(self, x: Any):
        out = list()
        for transform in self.transforms:
            out.append(transform(x))
        return tuple(out)


class TransformFixMatch(object):
    def __init__(self, mean, std, img_size=32, n=2, m=10):
        """Transformation for FixMatch.
        Callable returning (x_weak, x_strong)
        """
        self.weak = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(
                    size=img_size, padding=int(img_size * 0.125), padding_mode="reflect"
                ),
            ]
        )
        self.strong = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(
                    size=img_size, padding=int(img_size * 0.125), padding_mode="reflect"
                ),
                RandAugmentMCCutout(n=n, m=m),
            ]
        )
        self.normalize = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
        )

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class TransformFixMatchImageNet(TransformFixMatch):
    def __init__(self, mean, std, img_size=224, n=2, m=10, cut_rel=0.25, prob=1):
        if cut_rel == 0:
            RandAug = RandAugmentMC(n=n, m=m, prob=prob)
        else:
            RandAug = RandAugmentMCCutout(n=n, m=m, cut_rel=cut_rel, prob=prob)
        self.weak = transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
            ]
        )

        self.strong = transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
                RandAug,
            ]
        )
        self.normalize = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
        )


class TransformFixMatchISIC(TransformFixMatch):
    def __init__(self, mean, std, img_size=224, n=1, m=2, cut_rel=0.25, prob=1):
        if cut_rel == 0:
            RandAug = RandAugmentMC(n=n, m=m, prob=prob)
        else:
            RandAug = RandAugmentMCCutout(n=n, m=m, cut_rel=cut_rel, prob=prob)
        re_size = 300
        input_size = img_size
        self.weak = transforms.Compose(
            [
                transforms.Resize(re_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation([-180, 180]),
                transforms.RandomAffine(
                    [-180, 180], translate=[0.1, 0.1], scale=[0.7, 1.3]
                ),
                transforms.RandomCrop(input_size),
            ]
        )
        self.strong = transforms.Compose(
            [
                transforms.Resize(re_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                # transforms.ColorJitter(0.02, 0.02, 0.02, 0.01),
                transforms.RandomRotation([-180, 180]),
                transforms.RandomAffine(
                    [-180, 180], translate=[0.1, 0.1], scale=[0.7, 1.3]
                ),
                transforms.RandomCrop(input_size),
                RandAug,
            ]
        )
        self.normalize = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
        )


def seed_worker(worker_id):
    """
    https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    to fix https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/
    ensures different random numbers each batch with each worker every epoch while keeping reproducibility
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_split_indices(labels, num_labeled, num_val, _n_classes):
    """
    Split the train data into the following three set:
    (1) labeled data
    (2) unlabeled data
    (3) val data
    Data distribution of the three sets are same as which of the
    original training data.
    Inputs:
        labels: (np.int) array of labels
        num_labeled: (int)
        num_val: (int)
        _n_classes: (int)
    Return:
        the three indices for the three sets
    """
    # val_per_class = num_val // _n_classes
    val_indices = []
    train_indices = []

    num_total = len(labels)
    num_per_class = []
    for c in range(_n_classes):
        num_per_class.append((labels == c).sum().astype(int))

    # obtain val indices, data evenly drawn from each class
    for c, num_class in zip(range(_n_classes), num_per_class):
        val_this_class = max(int(num_val * (num_class / num_total)), 1)
        class_indices = np.where(labels == c)[0]
        np.random.shuffle(class_indices)
        val_indices.append(class_indices[:val_this_class])
        train_indices.append(class_indices[val_this_class:])

    # split data into labeled and unlabeled
    labeled_indices = []
    unlabeled_indices = []

    # num_labeled_per_class = num_labeled // _n_classes

    for c, num_class in zip(range(_n_classes), num_per_class):
        num_labeled_this_class = max(int(num_labeled * (num_class / num_total)), 1)
        labeled_indices.append(train_indices[c][:num_labeled_this_class])
        unlabeled_indices.append(train_indices[c][num_labeled_this_class:])

    labeled_indices = np.hstack(labeled_indices)
    unlabeled_indices = np.hstack(unlabeled_indices)
    val_indices = np.hstack(val_indices)

    return labeled_indices, unlabeled_indices, val_indices
