import os
from typing import Any, Dict, Iterable, Optional, Tuple, Union

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
)
from query.query_uncertainty import get_bald_fct, get_bay_entropy_fct
from utils.torch_utils import (
    AbstractBatchData,
    GetModelOutputs,
    get_batch_data,
    get_functional_from_loader,
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
        test_data = get_functional_from_loader(test_loader)

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
        pl_module,
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
        grid_unc = get_functional_from_loader(grid_loader, functions_unc)
        grid_data["xx"] = grid_arrays[0]
        grid_data["yy"] = grid_arrays[1]
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
        name_suffix = ""
        if epoch is not None:
            name_suffix = "_{:05d}".format(epoch)
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
        functions = (get_batch_data, GetModelOutputs(model, device=device))
        loader_dict = get_functional_from_loader(dataloader, functions)
        return loader_dict


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
        out_dict["entropy"] = entropy_fct(x)
        out_dict["bald"] = bald_fct(x)
        return out_dict
