# from data.data import TorchVisionDM
from copy import deepcopy
import os
import numpy as np
import hydra
from omegaconf import DictConfig

import matplotlib.pyplot as plt

from utils import config_utils
from utils.tensor import to_numpy
import utils
from trainer import ActiveTrainingLoop

from data.toy_dm import ToyDM
from toy_callback import ToyVisCallback
from query.storing import ActiveStore


def close_figs():
    plt.close("all")


@hydra.main(config_path="./config", config_name="config_toy")
def main(cfg: DictConfig):
    config_utils.print_config(cfg)
    train(cfg)


class ToyActiveLearningLoop(ActiveTrainingLoop):
    def init_callbacks(self):
        super().init_callbacks()
        if self.cfg.trainer.vis_callback:
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

    def active_callback(self) -> ActiveStore:
        """Executes the Active Callback from its parent class.
        Then updates internal dictionary with data from the pool and alongside queries.
        Creates Visualizations.

        Returns:
            ActiveStore: Carries infor regarding Queries.
        """
        active_store = super().active_callback()
        pool_loader = self.datamodule.pool_dataloader()
        pool_data = ToyVisCallback.get_outputs(self.model, pool_loader, self.device)
        # we loose ordering of queries by doing this!
        pool_data["queries"] = np.zeros(pool_data["data"].shape[0], dtype=bool)
        pool_data["queries"][active_store.requests] = 1
        active_store_dict = active_store.__dict__
        for key, val in active_store_dict.items():
            active_store_dict[key] = to_numpy(val)
        self.update_save_dict("active_store", active_store_dict)
        # pool_data is added here to internal _save_dict
        self.update_save_dict("pool_data", pool_data)

        # this only works if final_callback has been executed before active callback!
        train_data = self._save_dict["train_data"]
        val_data = self._save_dict["val_data"]
        grid_data = self._save_dict["grid"]

        # save_paths = (self.log_dir, utils.visuals_folder)
        save_paths = (self.log_dir,)

        # # obtain data from pool.
        # pool_set = self.datamodule.train_set.pool
        # acquired_data = []
        # for idx in active_store.requests:
        #     x, y = pool_set[idx]
        #     acquired_data.append(to_numpy(x))
        # acquired_data = np.array(acquired_data)

        grid_unc = deepcopy(self._save_dict["grid"])
        for key in train_data.keys():
            grid_unc.pop(key)
        ToyVisCallback.query_plot(
            train_data, val_data, grid_data, pool_data, grid_unc, save_paths, self.count
        )
        fig, axs = ToyVisCallback.fig_full_vis_2d([self._save_dict])
        fig.suptitle("Acquisition Step {}".format(self.count))
        fig.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "fig_full_vis.png"))
        ToyVisCallback.save_figs(
            save_paths=save_paths,
            save_name="fig_full_vis",
            savetype="png",
            name_suffix=ToyVisCallback.get_name_suffix(self.count),
        )
        return active_store


def train(cfg: DictConfig):
    active_dataset = cfg.active.num_labelled is not None
    utils.set_seed(cfg.trainer.seed)
    balanced = cfg.active.balanced
    num_classes = cfg.data.num_classes
    num_labelled = cfg.active.num_labelled

    datamodule = get_toy_dm(cfg, active_dataset)

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


def get_toy_dm(cfg: DictConfig, active_dataset: bool = True) -> ToyDM:
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
        persistent_workers=cfg.trainer.persistent_workers,
    )

    return datamodule


if __name__ == "__main__":
    main()
