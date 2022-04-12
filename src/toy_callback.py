import os
from typing import Any, Dict, Iterable, Optional, Tuple, Union, List

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from torch import Tensor
import torch.nn as nn

from data.toy_dm import ToyDM, make_toy_dataset
from plotlib.toy_plots import (
    close_figs,
    create_2d_grid_from_data,
    fig_class_full_2d,
    fig_uncertain_full_2d,
    vis_class_train_2d,
    vis_unc_train_2d,
    vis_class_val_2d,
)
from query.query_uncertainty import get_bald_fct, get_bay_entropy_fct, get_var_ratios
from utils.concat import (
    AbstractBatchData,
    GetClassifierOutputs,
    get_batch_data,
    concat_functional,
)
from models.abstract_classifier import AbstractClassifier


class ToyVisCallback(pl.Callback):
    def __init__(
        self, datamodule: ToyDM, save_paths: Union[str, Tuple[str]], device="cuda:0"
    ):
        """Callback Capturing the Logic for Visualizations.

        Args:
            datamodule (ToyDM): _description_
            save_paths (Union[str, Tuple[str]]): _description_
            device (str, optional): _description_. Defaults to "cuda:0".
        """
        self.device = device
        if isinstance(save_paths, str):
            save_paths = (save_paths,)
        self.save_paths = save_paths
        for save_path in save_paths:
            if os.path.isdir(save_path) is False:
                os.makedirs(save_path)
        self.datamodule = datamodule

        (
            self.train_loader,
            self.val_loader,
            self.test_loader,
            self.pool_loader,
            self.grid_loader,
            self.grid_arrays,
        ) = self.get_datamodule_loaders(datamodule)

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        unused: Optional[Any] = None,
    ) -> None:
        epoch = trainer.current_epoch

        save_paths = self.save_paths

        (
            train_data,
            val_data,
            test_data,
            pool_data,
            grid_data,
            grid_unc,
        ) = self.get_loaders_data(
            pl_module,
            self.train_loader,
            self.val_loader,
            self.test_loader,
            self.pool_loader,
            self.grid_loader,
            self.grid_arrays,
            self.device,
            test=False,
        )

        self.baseline_plots(
            train_data, val_data, grid_data, pool_data, grid_unc, save_paths, epoch
        )

        return super().on_train_epoch_end(trainer, pl_module, unused)

    # TODO: Put Data into better format -- this long tuple is not really workable!
    @staticmethod
    def get_datamodule_loaders(datamodule: ToyDM):
        """Get the Dataloaders needed for visualizations.

        Args:
            datamodule (ToyDM): Datamodule carrying the data.

        Returns:
            _type_: _description_
        """
        train_loader = datamodule.labeled_dataloader()
        val_loader = datamodule.val_dataloader()
        test_loader = datamodule.test_dataloader()
        try:
            pool_loader = datamodule.pool_dataloader()
        except TypeError:
            pool_loader = None

        # this stays only until better method for creating meshgrid is found!
        test_data = concat_functional(test_loader)

        grid_arrays = create_2d_grid_from_data(test_data["data"])
        X_grid = np.c_[grid_arrays[0].ravel(), grid_arrays[1].ravel()]
        grid_loader = datamodule.create_dataloader(
            make_toy_dataset(X_grid, np.zeros(X_grid.shape[0], dtype=np.int) - 1),
        )
        return (
            train_loader,
            val_loader,
            test_loader,
            pool_loader,
            grid_loader,
            grid_arrays,
        )

    @staticmethod
    def get_loaders_data(
        pl_module: pl.LightningModule,
        train_loader,
        val_loader,
        test_loader,
        pool_loader,
        grid_loader,
        grid_arrays,
        device="cuda:0",
        test=True,
    ):
        """Gives back the data in the format needed for visualizations when given dataloaders.
        Visualizations with ToyVisCallback.get_baseline_plots(returns)

        Args:
            pl_module (_type_): _description_
            train_loader (_type_): _description_
            val_loader (_type_): _description_
            test_loader (_type_): _description_
            pool_loader (_type_): _description_
            grid_loader (_type_): _description_
            grid_arrays (_type_): _description_
            device (str, optional): _description_. Defaults to "cuda:0".

        Returns:
            _type_: _description_
        """
        pl_module.eval()
        train_data = ToyVisCallback.get_outputs(pl_module, train_loader)
        val_data = ToyVisCallback.get_outputs(pl_module, val_loader)
        if test:
            test_data = ToyVisCallback.get_outputs(pl_module, test_loader)
        else:
            test_data = dict()
            for key in test_data.keys():
                test_data[key] = None
        grid_data = ToyVisCallback.get_outputs(pl_module, grid_loader)
        if pool_loader is not None:
            pool_data = ToyVisCallback.get_outputs(pl_module, pool_loader)
        else:
            pool_data = dict()
            for key in train_data.keys():
                pool_data[key] = None

        functions_unc = (GetModelUncertainties(pl_module, device=device),)
        grid_unc = concat_functional(grid_loader, functions_unc)
        grid_data["xx"] = grid_arrays[0]
        grid_data["yy"] = grid_arrays[1]
        pl_module.train()
        return train_data, val_data, test_data, pool_data, grid_data, grid_unc

    @staticmethod
    def baseline_plots(
        train_data: Dict[str, np.ndarray],
        val_data: Dict[str, np.ndarray],
        grid_data: Dict[str, np.ndarray],
        pool_data: Dict[str, np.ndarray],
        grid_unc: Dict[str, np.ndarray],
        save_paths: Tuple[str],
        epoch: Optional[int] = None,
        savetype="png",
    ):
        """Creates the baseline plots from the dictionaries obtained with get_outputs.

        Args:
            train_data (Dict[str, np.ndarray]): _description_
            val_data (Dict[str, np.ndarray]): _description_
            grid_data (Dict[str, np.ndarray]): _description_
            pool_data (Dict[str, np.ndarray]): _description_
            grid_unc (Dict[str, np.ndarray]): _description_
            save_paths (Tuple[str]): _description_
            epoch (Optional[int], optional): _description_. Defaults to None.
            savetype (str, optional): _description_. Defaults to "png".
        """
        name_suffix = ToyVisCallback.get_name_suffix(epoch)
        grid_arrays = (grid_data["xx"], grid_data["yy"])
        fig, axs = fig_class_full_2d(
            train_data["data"],
            val_data["data"],
            train_data["label"],
            val_data["label"],
            grid_lab=grid_data["pred"],
            grid_arrays=grid_arrays,
            pred_unlabelled=pool_data["data"],
        )
        # how to access possibility for functions?
        for save_path in save_paths:
            file_path = os.path.join(
                save_path, "fig_class_full_2d{}.{}".format(name_suffix, savetype)
            )
            plt.savefig(file_path)
        close_figs()

        fig, axs = fig_uncertain_full_2d(
            train_data["data"], train_data["label"], grid_unc, grid_arrays
        )
        for save_path in save_paths:
            file_path = os.path.join(
                save_path, "fig_uncertain_full_2d{}.{}".format(name_suffix, savetype)
            )
            plt.savefig(file_path)
        close_figs()

    @staticmethod
    def query_plot(
        train_data: Dict[str, np.ndarray],
        val_data: Dict[str, np.ndarray],
        grid_data: Dict[str, np.ndarray],
        pool_data: Dict[str, np.ndarray],
        grid_unc: Dict[str, np.ndarray],
        save_paths: Tuple[str],
        loop: Optional[int] = None,
        savetype="png",
    ):
        name_suffix = ToyVisCallback.get_name_suffix(loop)
        acquired_data = pool_data["data"][pool_data["queries"]]
        fig, axs = fig_class_full_2d(
            train_data["data"],
            val_data["data"],
            train_data["label"],
            val_data["label"],
            grid_lab=grid_data["pred"],
            grid_arrays=(grid_data["xx"], grid_data["yy"]),
            pred_unlabelled=pool_data["data"],
            pred_queries=acquired_data,
        )

        ToyVisCallback.save_figs(
            save_paths, "fig_uncertain_full_2d", savetype, name_suffix
        )
        close_figs()

    @staticmethod
    def save_figs(
        save_paths: Union[Path, str],
        save_name: str,
        savetype: Optional[str] = None,
        name_suffix="",
    ):
        if savetype is not None:
            for save_path in save_paths:
                file_path = os.path.join(
                    save_path,
                    f"{save_name}{name_suffix}.{savetype}",
                )
                plt.savefig(file_path)

    @staticmethod
    def get_name_suffix(count: int):
        name_suffix = ""
        if count is not None:
            name_suffix = "_{:05d}".format(count)
        return name_suffix

    @staticmethod
    def get_outputs(
        model: nn.Module, dataloader: Iterable, device: str = "cuda:0"
    ) -> Dict[str, np.ndarray]:
        """Obtain data and the output of the model of the dataloader.

        Args:
            model (nn.Module): Classification Model
            dataloader (Iterable): Dataloader
            device (str, optional): "cpu" or "cuda". Defaults to "cuda:0".

        Returns:
            Dict[str, np.ndarray]: Data, Model outputs and predictions
        """
        functions = (get_batch_data, GetClassifierOutputs(model, device=device))
        loader_dict = concat_functional(dataloader, functions)
        return loader_dict

    @staticmethod
    def plot_full_vis_axis(
        ax: plt.axis, data_dict: dict, x: int, y: int, keys: list, offset: int
    ):
        """Add the Plots for the full_vis.png plots given the axis and current location (x, y).

        Args:
            ax (plt.axis): axis on which plot is created
            data_dict (dict): carrying inputs from save_dict of one training
            x (int): plot grid, x
            y (int): plot grid, y
            keys (list): list of keys for uncertainties
            offset (int): how many plots are to be created before uncertainties with keys
        """
        grid_arrays = (data_dict["grid"]["xx"], data_dict["grid"]["yy"])
        if x == 0:
            ax.set_title("Training Data")
            ax = vis_class_train_2d(
                ax=ax,
                predictors=data_dict["train_data"]["data"],
                labels=data_dict["train_data"]["label"],
                grid_labels=data_dict["grid"]["pred"],
                grid_arrays=grid_arrays,
                predictors_unlabeled=data_dict["pool_data"]["data"],
            )
        if x == 1:
            ax.set_title("Acquisition Plot")
            pool_data = data_dict["pool_data"]
            query_data = pool_data["data"][pool_data["queries"]]
            ax = vis_class_train_2d(
                ax=ax,
                predictors=data_dict["train_data"]["data"],
                labels=data_dict["train_data"]["label"],
                grid_labels=data_dict["grid"]["pred"],
                grid_arrays=grid_arrays,
                predictors_unlabeled=data_dict["pool_data"]["data"],
                predictors_query=query_data,
            )
        if x == 2:
            ax.set_title("Validation Data")
            ax = vis_class_val_2d(
                ax=ax,
                predictors=data_dict["val_data"]["data"],
                labels=data_dict["val_data"]["label"],
                grid_labels=data_dict["grid"]["pred"],
                grid_arrays=grid_arrays,
            )
        if x >= offset:
            index = x - offset
            # print("Index", index)
            key = keys[index]
            ax.set_title("Uncertainty Map: {}".format(key))
            if key == "variationratios":
                contourf_kwargs = dict(vmin=0, vmax=1)
            else:
                contourf_kwargs = 0
            ax = vis_unc_train_2d(
                ax=ax,
                predictors=data_dict["train_data"]["data"],
                labels=data_dict["train_data"]["label"],
                grid_arrays=grid_arrays,
                grid_labels=data_dict["grid"][key],
            )

    @staticmethod
    def fig_full_vis_2d(listdicts: List[dict]):

        num_rows = len(listdicts)

        grid_unc: dict = listdicts[0]["grid"]
        keys = [
            key
            for key in grid_unc.keys()
            if key not in ["data", "label", "pred", "xx", "yy", "prob"]
        ]
        # print(keys)
        offset = 3
        num_cols = offset + len(keys)
        # num_cols = 5
        num_plots = num_rows * num_cols

        fig, axs = plt.subplots(
            num_rows,
            num_cols,
            sharex="col",
            sharey="row",
            squeeze=False,
            figsize=(num_cols * 3, num_rows * 3),
        )
        for y, ax_row in enumerate(axs):
            data_dict = listdicts[y]
            for x, ax in enumerate(ax_row):
                count = x + num_cols * y
                if count >= num_plots:
                    continue
                ToyVisCallback.plot_full_vis_axis(ax, data_dict, x, y, keys, offset)
        return fig, axs


class GetModelUncertainties(AbstractBatchData):
    def __init__(self, model: AbstractClassifier, device="cuda:0"):
        """Callable class to obtain uncertainties.

        Args:
            model (AbstractClassifier): Classifier with MC possibility.
            device (str, optional): _description_. Defaults to "cuda:0".
        """
        super().__init__()
        self.model = model
        self.device = device
        self.model = self.model.to(device)

    def custom_call(self, x: Tensor, out_dict: dict, **kwargs):
        x = x.to(self.device)
        bald_fct = get_bald_fct(self.model)
        entropy_fct = get_bay_entropy_fct(self.model)
        varratios_fct = get_var_ratios(self.model)
        out_dict["entropy"] = entropy_fct(x)
        out_dict["bald"] = bald_fct(x)
        out_dict["varratios"] = varratios_fct(x)
        return out_dict
