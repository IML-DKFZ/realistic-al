from typing import Callable

from loguru import logger
from torch.utils.data import DataLoader

from .data import TorchVisionDM
from .transformations import get_transform
from .utils import (
    MultiHeadedTransform,
    RandomFixedLengthSampler,
    TransformFixMatch,
    TransformFixMatchImageNet,
    TransformFixMatchISIC,
    activesubset_from_subset,
    seed_worker,
)


def fixmatch_train_dataloader(dm: TorchVisionDM, mu: int, min_samples: int = 6400):
    """Returns the Concatenated Daloader used for FixMatch Training given the datamodule"""
    train_pool = activesubset_from_subset(dm.train_set.pool)
    if isinstance(dm, TorchVisionDM):
        if "isic" in dm.dataset:
            train_pool.transform = TransformFixMatchISIC(
                mean=dm.mean, std=dm.std, n=2, m=10, cut_rel=0, prob=1
            )
        elif dm.dataset == "miotcd":
            train_pool.transform = TransformFixMatchImageNet(
                mean=dm.mean, std=dm.std, n=2, m=10, cut_rel=0, prob=1
            )
        else:
            train_pool.transform = TransformFixMatch(
                mean=dm.mean, std=dm.std, n=2, m=10
            )
    elif isinstance(dm, ToyDM):
        train_pool.transform = MultiHeadedTransform(
            [
                get_transform(name="toy_identity", mean=dm.mean, std=dm.std),
                get_transform(name="toy_gauss_0.05", mean=dm.mean, std=dm.std),
            ]
        )
    else:
        raise NotImplementedError(
            "For DataModule of Type {} no FixMatch Transformation is implemented.".format(
                type(dm)
            )
        )

    # Keep amount of workers fixed for training.
    if dm.num_workers > 2:
        workers_sup = max(2, (dm.num_workers) // (mu + 1))
    else:
        workers_sup = 0
    workers_sem = dm.num_workers - workers_sup
    # print("Workers Semi={}".format(workers_sem))
    if len(train_pool) < dm.batch_size * mu:
        sem_loader = DataLoader(
            train_pool,
            batch_size=dm.batch_size * mu,
            num_workers=workers_sem,
            sampler=RandomFixedLengthSampler(train_pool, min_samples * mu),
            pin_memory=dm.pin_memory,
            drop_last=True,
            worker_init_fn=seed_worker,
            persistent_workers=dm.persistent_workers,
            timeout=dm.timeout,
        )
    else:
        sem_loader = DataLoader(
            train_pool,
            batch_size=dm.batch_size * mu,
            num_workers=workers_sem,
            shuffle=True,
            pin_memory=dm.pin_memory,
            drop_last=True,
            worker_init_fn=seed_worker,
            persistent_workers=dm.persistent_workers,
            timeout=dm.timeout,
        )

    # Increase size of small datasets to make use of multiple workers
    # and limit the amount of dataloader reinits in concat dataloader
    sample_trainset = len(dm.train_set)
    len_sem_loader = len(sem_loader)
    if sample_trainset // dm.batch_size < len_sem_loader:
        resample_size = len_sem_loader * dm.batch_size

        resample_size = max(min_samples, resample_size)
        sup_loader = DataLoader(
            dm.train_set,
            batch_size=dm.batch_size,
            sampler=RandomFixedLengthSampler(dm.train_set, resample_size),
            num_workers=dm.num_workers,
            pin_memory=dm.pin_memory,
            drop_last=dm.drop_last,
            worker_init_fn=seed_worker,
            persistent_workers=dm.persistent_workers,
            timeout=dm.timeout,
        )
    else:
        sup_loader = DataLoader(
            dm.train_set,
            batch_size=dm.batch_size,
            shuffle=dm.shuffle,
            num_workers=dm.num_workers,
            pin_memory=dm.pin_memory,
            drop_last=dm.drop_last,
            worker_init_fn=seed_worker,
            persistent_workers=dm.persistent_workers,
            timeout=dm.timeout,
        )
    return [sup_loader, sem_loader]


def wrap_fixmatch_train_dataloader(
    dm: TorchVisionDM, mu: int
) -> Callable[[], DataLoader]:
    """Returns the executable function which allows to obtain the fixmatch train_dataloaders."""

    def train_dataloader():
        return fixmatch_train_dataloader(dm, mu)

    return train_dataloader
