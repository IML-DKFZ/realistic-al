from torch.utils.data import DataLoader
from copy import deepcopy

from .utils import (
    ConcatDataloader,
    TransformFixMatch,
    activesubset_from_subset,
    seed_worker,
)
from .data import TorchVisionDM


def fixmatch_train_dataloader(dm: TorchVisionDM, mu: int):
    """Returns the Concatenated Daloader used for FixMatch Training given the datamodule"""
    train_pool = activesubset_from_subset(dm.train_set.pool._dataset)
    train_pool.transform = TransformFixMatch(mean=dm.mean, std=dm.std)
    # Keep amount of workers fixed for training.
    # workers_sup = max(0, (dm.num_workers * 1) // 3)
    # workers_sem = max(0, (dm.num_workers * 2) // 3)
    workers_sup = dm.num_workers
    workers_sem = dm.num_workers

    return ConcatDataloader(
        DataLoader(
            dm.train_set,
            batch_size=dm.batch_size * mu,
            num_workers=workers_sup,
            shuffle=True,
            pin_memory=dm.pin_memory,
            drop_last=dm.drop_last,
            worker_init_fn=seed_worker,
        ),
        DataLoader(
            train_pool,
            batch_size=dm.batch_size * mu,
            num_workers=workers_sem,
            shuffle=True,
            pin_memory=dm.pin_memory,
            drop_last=True,
            worker_init_fn=seed_worker,
        ),
    )


def wrap_fixmatch_train_dataloader(dm: TorchVisionDM, mu: int):
    """Returns the executable function which allows to obtain the fixmatch train_dataloaders."""

    def train_dataloader():
        return fixmatch_train_dataloader(dm, mu)

    return train_dataloader
