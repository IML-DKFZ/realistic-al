import math
import os
from typing import Callable

import hydra
import numpy as np
from omegaconf import DictConfig

import utils
from data.data import TorchVisionDM
from trainer import ActiveTrainingLoop
from run_training import get_torchvision_dm
from utils import config_utils


@hydra.main(config_path="./config", config_name="config")
def main(cfg: DictConfig):
    config_utils.print_config(cfg)
    utils.set_seed(cfg.trainer.seed)

    active_loop(
        cfg,
        ActiveTrainingLoop,
        get_torchvision_dm,
        cfg.active.num_labelled,
        cfg.active.balanced,
        cfg.active.acq_size,
        cfg.active.num_iter,
    )


def active_loop(
    cfg: DictConfig,
    ActiveTrainingLoop=ActiveTrainingLoop,
    get_active_dm_from_config: Callable = get_torchvision_dm,
    num_labelled: int = 100,
    balanced: bool = True,
    acq_size: int = 10,
    num_iter: int = 0,
):
    datamodule = get_active_dm_from_config(cfg)
    num_classes = cfg.data.num_classes
    if balanced:
        datamodule.train_set.label_balanced(
            n_per_class=num_labelled // num_classes, num_classes=num_classes
        )
    else:
        datamodule.train_set.label_randomly(num_labelled)

    if num_iter == 0:
        num_iter = math.ceil(len(datamodule.train_set) / acq_size)

    active_stores = []
    for i in range(num_iter):
        # Perform active learning iteration with training and labeling
        trainer = ActiveTrainingLoop(
            cfg, count=i, datamodule=datamodule, base_dir=os.getcwd()
        )
        trainer.main()
        active_store = trainer.active_callback()
        datamodule.train_set.label(active_store.requests)
        active_stores.append(active_store)

    val_accs = np.array([active_store.accuracy_val for active_store in active_stores])
    test_accs = np.array([active_store.accuracy_test for active_store in active_stores])
    num_samples = np.array([active_store.n_labelled for active_store in active_stores])
    add_labels = np.stack(
        [active_store.labels for active_store in active_stores], axis=0
    )
    store_path = "."

    # This can be deleted!
    if True:
        import matplotlib.pyplot as plt

        plt.clf()
        plt.plot(num_samples, val_accs)
        plt.savefig(os.path.join(store_path, "val_accs_vs_num_samples.pdf"))
        plt.clf()
        plt.plot(num_samples, test_accs)
        plt.savefig(os.path.join(store_path, "test_accs_vs_num_samples.pdf"))
        plt.clf()

    np.savez(
        os.path.join(store_path, "stored.npz"),
        val_acc=val_accs,
        test_acc=test_accs,
        num_samples=num_samples,
        added_labels=add_labels,
    )


if __name__ == "__main__":
    main()
