import os

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl

from typing import Optional, Tuple

from . import query_diversity, query_uncertainty

from query.storing import ActiveStore
from plotlib import active_plots
import matplotlib.pyplot as plt

# TODO: simplify the logic of this class -- I do not 100% understand this anymore!
class QuerySampler:
    def __init__(
        self, cfg, model: nn.Module, count: Optional[int] = None, device="cuda:0"
    ):
        """Carries functionality to query samples from the pool.
        acq_size is selected based on cfg.active.acq_size


        Args:
            cfg (_type_): carries needed parameters
            model (nn.Module): _description_
            count (Optional[int], optional): used for vis -- which iteration. Defaults to None.
            device (str, optional): _description_. Defaults to "cuda:0".
        """
        self.model = model
        self.cfg = cfg
        self.count = count
        self.device = device
        self.m = cfg.active.m
        self.acq_size = cfg.active.acq_size
        self.acq_method = cfg.query.name

    def query_samples(
        self, datamodule: pl.LightningDataModule
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Query samples with the selected Query Sampler for the Active Datamodule

        Args:
            datamodule (pl.LightningDataModule): _description_

        Returns:
            Tuple[np.ndarray, np.ndarray]: Queries -- (rankings + pool_indices)
        """
        # possibility to select random subset of pool with certain Size via parameter m
        pool_loader = datamodule.pool_dataloader(batch_size=64, m=self.m)

        # Core Set uses test transformations for the labeled set.
        # Own results indicate that there is no difference in performance
        # labeled_loader = datamodule.train_dataloader() # This is deprecated, CoreSet uses Test time transforms for labeled data
        labeled_loader = datamodule.labeled_dataloader(batch_size=64)

        acq_inds, acq_vals = self.ranking_step(pool_loader, labeled_loader)
        acq_inds = datamodule.get_pool_indices(acq_inds)
        return acq_inds, acq_vals

    def active_callback(self, datamodule: pl.LightningDataModule) -> ActiveStore:
        """Queries samples on the pool of the datamodule with selected method, evaluates the current model.
        Requests are the indices to be labelled relative to the pool. (This changes if pool changes)

        Args:
            datamodule (pl.LightningDataModule): _description_

        Returns:
            ActiveStore: _description_
        """
        # TODO: Make this use pre-defined methods for the model!
        acq_inds, acq_vals = self.query_samples(datamodule)

        acq_data, acq_labels = obtain_data_from_pool(
            datamodule.train_set.pool, acq_inds
        )
        n_labelled = datamodule.train_set.n_labelled
        accuracy_val = evaluate_accuracy(self.model, datamodule.val_dataloader())
        accuracy_test = evaluate_accuracy(self.model, datamodule.test_dataloader())

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
        self, pool_loader, labeled_loader
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes Ranking of the data and returns indices of
        data to be acquired (with scores if possible).

        Acquisition Strategy: Values with highest scores are acquired.

        Args:
            pool_loader (_type_): _description_
            labeled_loader (_type_): _description_

        Returns:
            indices, scores: indices of the pool and scores for acquisition
        """
        self.model = self.model.to(self.device)
        self.model.eval()
        acq_size = self.acq_size
        if self.acq_method.split("_")[0] in query_uncertainty.names:
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
        elif self.acq_method.split("_")[0] in query_diversity.names:
            acq_ind, acq_scores = query_diversity.query_sampler(
                self.cfg, self.model, labeled_loader, pool_loader, acq_size=acq_size
            )

        else:
            raise NotImplementedError()

        return acq_ind, acq_scores


def obtain_data_from_pool(pool, indices):
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


def evaluate_accuracy(model, dataloader):
    if dataloader is None:
        return 0
    counts = 0
    correct = 0
    for batch in dataloader:
        x, y = batch
        x = x.to("cuda:0")
        out = model(x)
        pred = torch.argmax(out, dim=1)
        correct += (pred.cpu() == y).sum().item()
        counts += y.shape[0]
    return correct / counts


def vis_callback(n_labelled, acq_labels, acq_data, acq_vals, num_classes, count=None):
    suffix = ""
    if count is not None:
        suffix = f"_{count}"
    vis_path = "."
    fig, axs = active_plots.visualize_samples(
        active_plots.normalize(acq_data), acq_vals
    )
    plt.savefig(os.path.join(vis_path, "labeled_samples{}.pdf".format(suffix)))
    plt.clf()

    fig, axs = active_plots.visualize_labels(acq_labels, num_classes)
    plt.savefig(os.path.join(vis_path, "labelled_targets{}.pdf".format(suffix)))
    plt.clf()
