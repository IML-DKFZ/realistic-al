# from data.data import TorchVisionDM
from abc import abstractclassmethod
import os
from collections import defaultdict
from typing import Any, Callable, Optional, Tuple
from importlib_metadata import entry_points
import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from torch import Tensor
from query.query_uncertainty import get_bald_fct, get_bay_entropy_fct
from utils import config_utils

import utils
from trainer import ActiveTrainingLoop

from data.toy_dm import ToyDM, make_toy_dataset
from plotlib.toy_plots import (
    create_2d_grid_from_data,
    fig_class_full_2d,
    fig_uncertain_full_2d,
)

active_dataset = True


@hydra.main(config_path="./config", config_name="config_toy")
def main(cfg: DictConfig):
    config_utils.print_config(cfg)
    train(cfg)


class ToyActiveLearningLoop(ActiveTrainingLoop):
    def final_callback(self):
        # get data for training
        self.model = self.model.to(self.device)
        self.model.eval()
        try:
            pool_loader = self.datamodule.pool_dataloader()
            pool_data = get_outputs(self.model, pool_loader, self.device)
        except TypeError:
            pool_data = None

        # keep in mind to keep training data without transformations here!
        train_loader = self.datamodule.labeled_dataloader()
        train_data = get_outputs(self.model, train_loader, self.device)
        # get data for validation
        val_loader = self.datamodule.val_dataloader()
        val_data = get_outputs(self.model, val_loader, self.device)
        # get data for test (optional)
        test_loader = self.datamodule.test_dataloader()
        test_data = get_outputs(self.model, test_loader, self.device)

        # get data for grid

        grid = create_2d_grid_from_data(test_data["data"])
        X_grid = np.c_[grid[0].ravel(), grid[1].ravel()]
        grid_loader = self.datamodule.create_dataloader(
            make_toy_dataset(X_grid, np.zeros(X_grid.shape[0], dtype=np.int) - 1),
        )

        grid_data = get_outputs(self.model, grid_loader, self.device)
        grid_data["xx"] = grid[0]
        grid_data["yy"] = grid[1]

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
            grid_arrays=grid,
            pred_unlabelled=pred_unlabelled,
        )
        import matplotlib.pyplot as plt
        import os

        plt.savefig(f"{self.log_dir}/fig_class_full_2d.png")

        plt.savefig(f"{utils.visuals_folder}/fig_class_full_2d.png")
        plt.clf()
        plt.cla()

        functions_unc = (
            # get_batch_data,
            GetModelUncertainties(self.model, device=self.device),
        )
        grid_unc = get_functional_from_loader(grid_loader, functions_unc)
        fig, axs = fig_uncertain_full_2d(
            train_data["data"], train_data["label"], grid_unc, grid
        )

        plt.savefig(f"{self.log_dir}/fig_uncertain_full_2d.png")

        plt.savefig(f"{utils.visuals_folder}/fig_uncertain_full_2d.png")
        plt.clf()
        plt.cla()
        self.update_save_dict("train_data", train_data)
        self.update_save_dict("val_data", val_data)
        self.update_save_dict("test_data", test_data)

        self.update_save_dict("grid", grid_data)

    # TODO: This is just for prototyping -- CLEAN THIS UP ASAP!
    def active_callback(self):
        active_store = super().active_callback()
        self.model.eval()
        self.model = self.model.to(self.device)
        # TODO: Try except might be unnecessary here -- check again.
        try:
            pool_loader = self.datamodule.pool_dataloader()
            pool_data = get_outputs(self.model, pool_loader, self.device)
            self.update_save_dict("pool_data", pool_data)
        except TypeError:
            pool_data = None

        # keep in mind to keep training data without transformations here!
        # train_loader = self.datamodule.labeled_dataloader()
        # train_data = get_outputs(self.model, train_loader, self.device)
        # get data for validation
        # val_loader = self.datamodule.val_dataloader()
        # val_data = get_outputs(self.model, val_loader, self.device)
        # get data for test (optional)
        # test_loader = self.datamodule.test_dataloader()
        # test_data = get_outputs(self.model, test_loader, self.device)

        # # get data for grid
        # grid = create_2d_grid_from_data(test_data["data"])
        # X_grid = np.c_[grid[0].ravel(), grid[1].ravel()]
        # grid_loader = self.datamodule.create_dataloader(
        #     make_toy_dataset(X_grid, np.zeros(X_grid.shape[0], dtype=np.int) - 1),
        # )
        # grid_data = get_outputs(self.model, grid_loader, self.device)

        # this only works if final_callback has been executed before active callback!
        train_data = self._save_dict["train_data"]
        val_data = self._save_dict["val_data"]
        grid_data = self._save_dict["grid"]
        grid_arrays = (self._save_dict["grid"]["xx"], self._save_dict["grid"]["yy"])

        pool_set = self.datamodule.train_set.pool
        acquired_data = []
        for idx in active_store.requests:
            x, y = pool_set[idx]
            acquired_data.append(to_numpy(x))
        acquired_data = np.array(acquired_data)

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
            grid_arrays=grid_arrays,
            pred_unlabelled=pred_unlabelled,
            pred_queries=acquired_data,
        )
        import matplotlib.pyplot as plt

        plt.savefig(f"{self.log_dir}/fig_class_full_2d-active.png")

        plt.savefig(f"{utils.visuals_folder}/fig_class_full_2d-active.png")
        plt.clf()
        plt.cla()
        return active_store


def get_outputs_deprecated(model, dataloader, device="cuda:0"):
    """Gets the data, labels, softmax and predictions of model for the dataloader"""
    full_outputs = defaultdict(list)
    for x, y in dataloader:
        full_outputs["data"].append(to_numpy(x))
        x = x.to(device)
        out = model(x)
        full_outputs["prob"].append(to_numpy(torch.exp(out)))
        # get max class
        pred = torch.argmax(out, dim=1)
        full_outputs["pred"].append(to_numpy(pred))
        if y is not None:
            full_outputs["label"].append(to_numpy(y))
        else:
            # create dummy label with same shape as predictions if no labels applicable
            # value of -1
            dummy_label = torch.ones_like(pred) * -1
            full_outputs["label"].append(to_numpy(dummy_label))
    # convert everyting to one big numpy array
    for key, value in full_outputs.items():
        full_outputs[key] = np.concatenate(value)
    return full_outputs


def get_outputs(model, dataloader, device="cuda:0"):
    functions = (get_batch_data, GetModelOutputs(model, device=device))
    loader_dict = get_functional_from_loader(dataloader, functions)
    return loader_dict


def get_batch_data(batch: Any, out_dict: dict = None) -> dict:
    """Access data inside of a batch and return it in form of a dictionary.
    If no out_dict is given then, a new dict is created and returned.

    Args:
        batch (Any): Batch from a dataloader
        out_dict (dict, optional): dictionary for extension. Defaults to None.

    Raises:
        NotImplemented: _description_

    Returns:
        dict: dictionary carrying batch data
    """
    if out_dict is None:
        out_dict = dict()
    if isinstance(batch, (list, tuple)):
        x, y = batch
    else:
        raise NotImplemented(
            "Currently this function is only implemented for batches of type tuple"
        )
    out_dict["data"] = to_numpy(x)
    if y is not None:
        out_dict["label"] = to_numpy(y)
    else:
        # create dummy label with same shape as predictions if no labels applicable
        # value of -1
        dummy_label = torch.ones(x.shape[0], dtype=torch.long) * -1
        out_dict["label"] = to_numpy(dummy_label)
    return out_dict


class AbstractBatchData(object):
    def __init__(self):
        pass

    def __call__(self, batch, out_dict: dict = None):
        if out_dict is None:
            out_dict = dict()
        if isinstance(batch, (list, tuple)):
            x, y = batch
        else:
            raise NotImplemented(
                "Currently this function is only implemented for batches of type tuple"
            )
        out_dict = self.custom_call(x, out_dict=out_dict, batch=batch, y=y)
        return out_dict

    @abstractclassmethod
    def custom_call(self, x: Tensor, out_dict: dict, **kwargs):
        raise NotImplementedError


class GetModelOutputs(AbstractBatchData):
    def __init__(self, model, device="cuda:0"):
        super().__init__()
        self.model = model
        self.device = device
        self.model = self.model.to(device)

    def custom_call(self, x: Tensor, out_dict: dict, **kwargs):
        x = x.to(self.device)
        with torch.no_grad():
            out = self.model(x)
            out_dict["prob"] = to_numpy(out)
            pred = torch.argmax(out, dim=1)
            out_dict["pred"] = to_numpy(pred)
        return out_dict


class GetModelUncertainties(AbstractBatchData):
    def __init__(self, model, device="cuda:0"):
        super().__init__()
        self.model = model
        self.device = device
        self.model = self.model.to(device)

    def custom_call(self, x: Tensor, out_dict: dict, **kwargs):
        x = x.to(self.device)
        bald_fct = get_bald_fct(self.model)
        entropy_fct = get_bay_entropy_fct(self.model)
        out_dict["entropy"] = to_numpy(entropy_fct(x))
        out_dict["bald"] = to_numpy(bald_fct(x))
        return out_dict


# this function might also work based on an abstract class (see GetModelOutputs)
def get_functional_from_loader(
    dataloader, functions: Tuple[Callable] = (get_batch_data)
):
    """Functions in functions are used on every batch in the dataloader.
    For a template regarding a function see `get_batch_data`
    Returns a dictionary"""
    loader_dict = defaultdict(list)
    for batch in dataloader:
        batch_dict = None
        for function in functions:
            batch_dict = function(batch, out_dict=batch_dict)
        for key, val in batch_dict.items():
            loader_dict[key].append(val)
    # create new dictionary so as to keep loader_dict as list!
    out_loader_dict = dict()
    for key, value in loader_dict.items():
        out_loader_dict[key] = np.concatenate(value)
    return out_loader_dict


def to_numpy(data: Any) -> np.ndarray:
    """Change data carrier data to a numpy array

    Args:
        data (Any): data carrier (torch, list, tuple, numpy)

    Raises:
        TypeError: _description_

    Returns:
        np.ndarray: array carrying data from data
    """
    if torch.is_tensor(data):
        return data.to("cpu").detach().numpy()
    elif isinstance(data, (tuple, list)):
        return np.array(data)
    elif isinstance(data, np.ndarray):
        return data
    else:
        raise TypeError(
            "Object data of type {} cannot be converted to np.ndarray".format(data.type)
        )


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
        cfg, datamodule, active=False, base_dir=os.getcwd()
    )
    training_loop.main()

    if active_dataset:
        active_store = training_loop.active_callback()
        dataset_indices = np.array(
            datamodule.train_set._oracle_to_pool_index(active_store.requests.tolist())
        )
        pool_set = datamodule.train_set.pool
        acquired_data = []
        for idx in active_store.requests:
            x, y = pool_set[idx]
            acquired_data.append(x.numpy())
        acquired_data = np.array(acquired_data)

        # datamodule.train_set.label(active_store.requests)
        # active_stores.append(active_store)


if __name__ == "__main__":
    main()
