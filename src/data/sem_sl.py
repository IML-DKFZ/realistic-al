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
    train_pool = activesubset_from_subset(dm.train_set.pool._dataset)
    train_pool.transform = TransformFixMatch(mean=dm.mean, std=dm.std)
    return ConcatDataloader(
        DataLoader(
            dm.train_set,
            batch_size=dm.batch_size * mu,
            num_workers=dm.num_workers,
            shuffle=True,
            pin_memory=dm.pin_memory,
            drop_last=dm.drop_last,
            worker_init_fn=seed_worker,
        ),
        DataLoader(
            train_pool,
            batch_size=dm.batch_size * mu,
            num_workers=dm.num_workers,
            shuffle=True,
            pin_memory=dm.pin_memory,
            drop_last=True,
            worker_init_fn=seed_worker,
        ),
    )


def wrap_fixmatch_train_dataloader(dm: TorchVisionDM, mu: int):
    def train_dataloader():
        return fixmatch_train_dataloader(dm, mu)

    return train_dataloader
