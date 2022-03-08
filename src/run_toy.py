# from data.data import TorchVisionDM
from abc import abstractclassmethod
import os
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union
from importlib_metadata import entry_points
import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from torch import Tensor
import pytorch_lightning as pl

import matplotlib.pyplot as plt

from query.query_uncertainty import get_bald_fct, get_bay_entropy_fct
from utils import config_utils
from utils.tensor import to_numpy
import utils
from trainer import ActiveTrainingLoop

from data.toy_dm import ToyDM, make_toy_dataset
from plotlib.toy_plots import (
    create_2d_grid_from_data,
    fig_class_full_2d,
    fig_uncertain_full_2d,
)
from toy_callback import ToyVisCallback
from utils.torch_utils import (
    AbstractBatchData,
    GetModelOutputs,
    get_batch_data,
    get_functional_from_loader,
)

active_dataset = False


def close_figs():
    plt.close("all")


@hydra.main(config_path="./config", config_name="config_toy")
def main(cfg: DictConfig):
    config_utils.print_config(cfg)
    train(cfg)


class ToyActiveLearningLoop(ActiveTrainingLoop):
    def init_callbacks(self):
        super().init_callbacks()
        if True:
            save_paths = [self.log_dir]
            if True:
                save_paths.append(utils.visuals_folder)
            save_paths = tuple(
                [os.path.join(save_path, "epoch_vis") for save_path in save_paths]
            )
            toy_vis_callback = ToyVisCallback(
                self.datamodule, save_paths=save_paths, device=self.device
            )
            self.callbacks.append(toy_vis_callback)

    def final_callback(self):
        # get data for training
        self.model = self.model.to(self.device)
        self.model.eval()

        # get loaders
        (
            train_loader,
            val_loader,
            test_loader,
            pool_loader,
            grid_loader,
            grid_arrays,
        ) = ToyVisCallback.get_datamodule_loaders(self.datamodule)

        (
            train_data,
            val_data,
            test_data,
            pool_data,
            grid_data,
            grid_unc,
        ) = ToyVisCallback.get_loaders_data(
            self.model,
            train_loader,
            val_loader,
            test_loader,
            pool_loader,
            grid_loader,
            grid_arrays,
        )
        # TODO: fix this before final commit
        save_paths = (self.log_dir, utils.visuals_folder)

        ToyVisCallback.baseline_plots(
            train_data, val_data, grid_data, pool_data, grid_unc, save_paths
        )

        # should pool_data also be added and if so, how handle Dict[str, None]?
        grid_data.update(grid_unc)
        self.update_save_dict("train_data", train_data)
        self.update_save_dict("val_data", val_data)
        self.update_save_dict("test_data", test_data)

        self.update_save_dict("grid", grid_data)

    # TODO: This is just for prototyping -- CLEAN THIS UP ASAP!
    def active_callback(self):
        active_store = super().active_callback()
        self.model.eval()
        self.model = self.model.to(self.device)

        pool_loader = self.datamodule.pool_dataloader()
        pool_data = ToyVisCallback.get_outputs(self.model, pool_loader, self.device)

        # pool_data is added here to the save_dict!
        self.update_save_dict("pool_data", pool_data)

        # this only works if final_callback has been executed before active callback!
        train_data = self._save_dict["train_data"]
        val_data = self._save_dict["val_data"]
        grid_data = self._save_dict["grid"]

        # obtain data from pool.
        pool_set = self.datamodule.train_set.pool
        acquired_data = []
        for idx in active_store.requests:
            x, y = pool_set[idx]
            acquired_data.append(to_numpy(x))
        acquired_data = np.array(acquired_data)

        # create plots
        # TODO: put this into ToyVisCallback
        if pool_data is None:
            pred_unlabelled = None
        else:
            pred_unlabelled = pool_data["data"]

        fig, axs = fig_class_full_2d(
            train_data["data"],
            val_data["data"],
            train_data["label"],
            val_data["label"],
            grid_lab=grid_data["pred"],
            grid_arrays=(grid_data["xx"], grid_data["yy"]),
            pred_unlabelled=pred_unlabelled,
            pred_queries=acquired_data,
        )
        import matplotlib.pyplot as plt

        plt.savefig(f"{self.log_dir}/fig_class_full_2d-active.png")

        plt.savefig(f"{utils.visuals_folder}/fig_class_full_2d-active.png")
        plt.clf()
        plt.cla()
        return active_store


def train(cfg: DictConfig):
    utils.set_seed(cfg.trainer.seed)
    balanced = cfg.active.balanced
    num_classes = cfg.data.num_classes
    num_labelled = cfg.active.num_labelled

    datamodule = ToyDM(
        num_samples=cfg.data.num_samples,
        num_test_samples=cfg.data.num_test_samples,
        data_noise=cfg.data.noise,
        data_root=cfg.trainer.data_root,
        batch_size=cfg.trainer.batch_size,
        dataset=cfg.data.name,
        min_train=cfg.active.min_train,
        val_split=cfg.data.val_split,
        random_split=cfg.active.random_split,
        active=active_dataset,
        num_classes=cfg.data.num_classes,
        mean=cfg.data.mean,
        std=cfg.data.std,
        transform_train=cfg.data.transform_train,
        transform_test=cfg.data.transform_test,
        shape=cfg.data.shape,
        num_workers=cfg.trainer.num_workers,
        seed=cfg.trainer.seed,
    )

    num_classes = cfg.data.num_classes
    if active_dataset:
        if balanced:
            datamodule.train_set.label_balanced(
                n_per_class=num_labelled // num_classes, num_classes=num_classes
            )
        else:
            datamodule.train_set.label_randomly(num_labelled)

    training_loop = ToyActiveLearningLoop(
        cfg, datamodule, active=active_dataset, base_dir=os.getcwd()
    )
    training_loop.main()

    if active_dataset:
        active_store = training_loop.active_callback()
    training_loop.log_save_dict()


if __name__ == "__main__":
    main()
