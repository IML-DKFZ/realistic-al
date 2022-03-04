# from data.data import TorchVisionDM
import os
from collections import defaultdict
import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from torch import Tensor
from utils import config_utils
from torch.utils.data import TensorDataset, DataLoader

import utils
from trainer import ActiveTrainingLoop

from data.toy_dm import ToyDM, make_toy_dataset
from plotlib.toy_plots import create_2d_grid_from_data, fig_class_full_2d

active_dataset = True


@hydra.main(config_path="./config", config_name="config_toy")
def main(cfg: DictConfig):
    config_utils.print_config(cfg)
    train(cfg)


class ToyActiveLearningLoop(ActiveTrainingLoop):
    def final_callback(self):
        # get data for training
        self.model = self.model.to(self.device)
        try:
            pool_loader = self.datamodule.pool_dataloader()
            pool_data = get_outputs(self.model, pool_loader, self.device)
        except TypeError:
            pool_data = None

        # keep in mind to keep training data without transformations here!
        train_loader = self.datamodule.train_dataloader()
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

        ## TODO: Add uncertainties here!

        # do iterations over all datasets

        # create a final plot

    # TODO: This is just for prototyping -- CLEAN THIS UP ASAP!
    def active_callback(self):
        active_store = super().active_callback()
        self.model = self.model.to(self.device)
        try:
            pool_loader = self.datamodule.pool_dataloader()
            pool_data = get_outputs(self.model, pool_loader, self.device)
        except TypeError:
            pool_data = None

        # keep in mind to keep training data without transformations here!
        train_loader = self.datamodule.train_dataloader()
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
        pool_set = self.datamodule.train_set.pool
        acquired_data = []
        for idx in active_store.requests:
            x, y = pool_set[idx]
            acquired_data.append(x.numpy())
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
            grid_arrays=grid,
            pred_unlabelled=pred_unlabelled,
            pred_queries=acquired_data,
        )
        import matplotlib.pyplot as plt
        import os

        plt.savefig(f"{self.log_dir}/fig_class_full_2d-active.png")

        plt.savefig(f"{utils.visuals_folder}/fig_class_full_2d-active.png")
        plt.clf()
        plt.cla()
        return active_store


# TODO: generalize this so that this works with functions and/or saves more data -- see c817h repo for ideas
# similar functions acq_from_batch in query_uncertainty.py
def get_outputs(model, dataloader, device="cuda:0"):
    """Gets the data, labels, softmax and predictions of model for the dataloader"""
    full_outputs = defaultdict(list)
    for x, y in dataloader:
        full_outputs["data"].append(tensor_to_numpy(x))
        x = x.to(device)
        out = model(x)
        full_outputs["prob"].append(tensor_to_numpy(out))
        # get max class
        pred = torch.argmax(out, dim=1)
        full_outputs["pred"].append(tensor_to_numpy(torch.exp(pred)))
        if y is not None:
            full_outputs["label"].append(tensor_to_numpy(y))
        else:
            # create dummy label with same shape as predictions if no labels applicable
            # value of -1
            dummy_label = torch.ones_like(pred) * -1
            full_outputs["label"].append(tensor_to_numpy(dummy_label))
    # convert everyting to one big numpy array
    for key, value in full_outputs.items():
        full_outputs[key] = np.concatenate(value)
    return full_outputs


def tensor_to_numpy(tensor: Tensor):
    return tensor.to("cpu").detach().numpy()


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
