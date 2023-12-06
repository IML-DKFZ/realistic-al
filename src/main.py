import math
import os
import time
from typing import Callable

import hydra
import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import DictConfig

import utils
from data.base_datamodule import BaseDataModule
from run_training import get_torchvision_dm, label_active_dm
from trainer import ActiveTrainingLoop
from utils import config_utils
from utils.log_utils import setup_logger


@hydra.main(config_path="./config", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    setup_logger()
    logger.info("Start logging")
    config_utils.print_config(cfg)
    logger.info("Set seed")
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


@logger.catch
def active_loop(
    cfg: DictConfig,
    ActiveTrainingLoop=ActiveTrainingLoop,
    get_active_dm_from_config: Callable[
        [DictConfig, bool], BaseDataModule
    ] = get_torchvision_dm,
    num_labelled: int = 100,
    balanced: bool = True,
    acq_size: int = 10,
    num_iter: int = 0,
):
    """Perform Active Learning over multiple loops.

    Args:
        cfg (DictConfig): config from main
        ActiveTrainingLoop (Class, optional): Class that defines what happens during the training. Defaults to ActiveTrainingLoop.
        get_active_dm_from_config (Callable, optional): class returning a datamodule usable for active learning. Defaults to get_torchvision_dm.
        num_labelled (int, optional): starting budget for active learning. Defaults to 100.
        balanced (bool, optional): whether starting budget is drawn balanced. Defaults to True.
        acq_size (int, optional): query size in each active learning loop. Defaults to 10.
        num_iter (int, optional): number of active learning loops. Defaults to 0.
    """
    logger.info("Instantiating Datamodule")
    datamodule = get_active_dm_from_config(cfg)
    label_active_dm(cfg, num_labelled, balanced, datamodule)

    if num_iter == 0:
        num_iter = math.ceil(len(datamodule.train_set) / acq_size)

    active_stores = []
    metric_paths = []
    for i in range(num_iter):
        logger.info("Start Active Loop {}".format(i))
        # Perform active learning iteration with training and labeling
        training_loop = ActiveTrainingLoop(
            cfg, count=i, datamodule=datamodule, base_dir=os.getcwd()
        )
        logger.info("Start Training of Loop {}".format(i))
        training_loop.main()
        if training_loop.trainer.interrupted:
            return
        logger.info("Start Acquisition of Loop {}".format(i))
        active_store = training_loop.active_callback()
        datamodule.train_set.label(active_store.requests)
        active_stores.append(active_store)
        training_loop.log_save_dict()
        cfg.active.num_labelled += cfg.active.acq_size
        logger.info("Finalized Loop {}".format(i))
        metric_paths.append(training_loop.log_dir)
        del training_loop
        time.sleep(1)

    store_path = "."
    metrics_df = []
    for metric_path in metric_paths:
        # laod metrics from csv
        metric_df = pd.read_csv(os.path.join(metric_path, "metrics.csv"))
        # select metrics for test  data
        cols = [col for col in metric_df.columns if "test" in col]
        metric_df = metric_df.loc[:, cols]
        metric_dict = dict(metric_df.iloc[-1])
        metrics_df.append(metric_dict)
    metrics_df = pd.DataFrame(metrics_df)
    metrics_df.to_csv(os.path.join(store_path, "test_metrics.csv"))

    val_accs = np.array([active_store.accuracy_val for active_store in active_stores])
    test_accs = np.array([active_store.accuracy_test for active_store in active_stores])
    num_samples = np.array([active_store.n_labelled for active_store in active_stores])
    add_labels = np.stack(
        [active_store.labels for active_store in active_stores], axis=0
    )
    request_pool = np.array([active_store.requests for active_store in active_stores])

    np.savez(
        os.path.join(store_path, "stored.npz"),
        val_acc=val_accs,
        test_acc=test_accs,
        num_samples=num_samples,
        added_labels=add_labels,
        request_pool=request_pool,
    )
    logger.success("Active Loop was finalized")


if __name__ == "__main__":
    main()
