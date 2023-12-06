import os
from typing import Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from data.base_datamodule import BaseDataModule
from query.storing import ActiveStore

from . import query_diversity, query_uncertainty


class QuerySampler:
    def __init__(
        self,
        cfg: DictConfig,
        model: nn.Module,
        count: Optional[int] = None,
        device: str = "cuda:0",
    ):
        """Carries functionality to query samples from the pool.
        acq_size is selected based on cfg.active.acq_size


        Args:
            cfg (DictConfig): config
            model (nn.Module): Model based on which query is performed
            count (Optional[int], optional): used for vis -- which iteration. Defaults to None.
            device (str, optional): _description_. Defaults to "cuda:0".
        """
        self.cfg = cfg
        self.count = count
        self.device = device
        self.m = cfg.active.m
        self.acq_size = cfg.active.acq_size
        self.acq_method = cfg.query.name
        self.model = model
        self.model = self.model.to(self.device)
        self.model.eval()

    def query_samples(
        self, datamodule: BaseDataModule, fastrandom: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Query samples with the selected Query Sampler for the Active Datamodule

        Args:
            datamodule (BaseDataModule): contains labeled and unlabeled data.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Queries [pool_indices, rankvalue]
        """
        # this random does not require predicting the whole unlabeled set.
        if self.acq_method == "random" and fastrandom:
            acq_inds = np.random.choice(
                np.arange(len(datamodule.train_set.pool)),
                size=self.cfg.active.acq_size,
                replace=False,
            )
            acq_vals = np.arange(self.cfg.active.acq_size) * -1

        else:
            # possibility to select random subset of pool with certain Size via parameter m
            pool_loader = datamodule.pool_dataloader(
                batch_size=datamodule.batch_size, m=self.m
            )

            # Core Set uses test transformations for the labeled set.
            # Own results indicate that there is no difference in performance
            # labeled_loader = datamodule.train_dataloader() # This is deprecated, CoreSet uses Test time transforms for labeled data
            labeled_loader = datamodule.labeled_dataloader(
                batch_size=datamodule.batch_size
            )

            acq_inds, acq_vals = self.ranking_step(pool_loader, labeled_loader)
            acq_inds = datamodule.get_pool_indices(acq_inds)
        return acq_inds, acq_vals

    def active_callback(
        self, datamodule: BaseDataModule, vis: bool = False
    ) -> ActiveStore:
        """Queries samples on the pool of the datamodule with selected method, evaluates the current model.
        Requests are the indices to be labelled relative to the pool. (This changes if pool changes)

        Args:
            datamodule (BaseDataModule): Datamodule carrying both labeled and unlabeled data.

        Returns:
            ActiveStore: Carries output values.
        """
        acq_inds, acq_vals = self.query_samples(datamodule)

        acq_data, acq_labels = obtain_data_from_pool(
            datamodule.train_set.pool, acq_inds
        )
        n_labelled = datamodule.train_set.n_labelled
        accuracy_val = evaluate_accuracy(
            self.model, datamodule.val_dataloader(), device=self.device
        )
        accuracy_test = evaluate_accuracy(
            self.model, datamodule.test_dataloader(), device=self.device
        )

        if vis:
            try:
                vis_callback(
                    n_labelled,
                    acq_labels,
                    acq_data,
                    acq_vals,
                    datamodule.num_classes,
                    count=self.count,
                )
            except:
                print(
                    "No Visualization with vis_callback function possible! \n Trying to Continue"
                )

        return ActiveStore(
            requests=acq_inds,
            n_labelled=n_labelled,
            accuracy_val=accuracy_val,
            accuracy_test=accuracy_test,
            labels=acq_labels,
        )

    def setup(self):
        pass

    def ranking_step(
        self, pool_loader: DataLoader, labeled_loader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes Ranking of the data and returns indices of
        data to be acquired (with scores if possible).

        Acquisition Strategy: Values with highest scores are acquired.

        Args:
            pool_loader (DataLoader): unlabeled dataloader w.o. augmentations
            labeled_loader (DataLoader): labeled dataloader w.o. augmentations

        Returns:
            indices, scores: indices of the pool and scores for acquisition
        """
        acq_size = self.acq_size
        if self.acq_method.split("_")[0] in query_uncertainty.NAMES:
            acq_function = query_uncertainty.get_acq_function(self.cfg, self.model)
            post_acq_function = query_uncertainty.get_post_acq_function(
                self.cfg, device=self.device
            )
            acq_ind, acq_scores = query_uncertainty.query_sampler(
                pool_loader,
                acq_function,
                post_acq_function,
                acq_size=acq_size,
                device=self.device,
            )
        elif self.acq_method.split("_")[0] in query_diversity.NAMES:
            acq_ind, acq_scores = query_diversity.query_sampler(
                self.cfg, self.model, labeled_loader, pool_loader, acq_size=acq_size
            )

        else:
            raise NotImplementedError()

        return acq_ind, acq_scores


def obtain_data_from_pool(pool: Dataset, indices: Iterable[int]):
    data, labels = [], []
    for ind in indices:
        sample = pool[ind]
        data.append(sample[0])
        labels.append(sample[1])
    data = torch.stack(data, dim=0)
    labels = torch.tensor(labels, dtype=torch.int)
    labels = labels.numpy()
    data = data.numpy()
    return data, labels


def evaluate_accuracy(model: torch.nn.Module, dataloader: DataLoader, device="cuda:0"):
    if dataloader is None:
        return 0
    counts = 0
    correct = 0
    for batch in dataloader:
        with torch.no_grad():
            x, y = batch
            x = x.to(device)
            out = model(x)
            pred = torch.argmax(out, dim=1)
            correct += (pred.cpu() == y).sum().item()
            counts += y.shape[0]
    return correct / counts
