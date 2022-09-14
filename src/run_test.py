from gc import callbacks
import math
import os
from typing import Callable, Union

from utils.io import load_omega_conf
import hydra
import numpy as np
from omegaconf import DictConfig
from models.callbacks.metrics_callback import ImbClassMetricCallback

import utils
from data.data import TorchVisionDM
from trainer import ActiveTrainingLoop
from run_training import get_torchvision_dm
from utils import config_utils
import time
from loguru import logger
from utils.log_utils import setup_logger

from utils.tensor import to_numpy

import pandas as pd

from pathlib import Path


def main(cfg: DictConfig, path: str):
    ckpt_path = Path(path) / "checkpoints"
    checkpoints = []
    for ckpt in ckpt_path.iterdir():
        checkpoints.append(ckpt)
    if len(checkpoints) == 0:
        logger.info("There is no checkpoint")
    if len(checkpoints) > 1:
        raise NotImplementedError
    ckpt_path = checkpoints[0]

    monitor_callback = 2

    setup_logger(path=path)
    logger.info("Start logging")
    config_utils.print_config(cfg)
    logger.info("Set seed")
    utils.set_seed(cfg.trainer.seed)

    num_labelled = cfg.active.num_labelled
    balanced = cfg.active.balanced
    acq_size = cfg.active.acq_size
    num_iter = cfg.active.num_iter

    logger.info("Instantiating Datamodule")
    datamodule = get_torchvision_dm(cfg)
    num_classes = cfg.data.num_classes
    if cfg.data.name == "isic2019" and balanced:
        label_balance = 40
        datamodule.train_set.label_balanced(
            n_per_class=label_balance // num_classes, num_classes=num_classes
        )
        label_random = num_labelled - label_balance
        if label_random > 0:
            datamodule.train_set.label_randomly(label_random)
    elif datamodule.imbalance and balanced:
        label_balance = cfg.data.num_classes * 5
        datamodule.train_set.label_balanced(
            n_per_class=label_balance // num_classes, num_classes=num_classes
        )
        label_random = num_labelled - label_balance
        if label_random > 0:
            datamodule.train_set.label_randomly(label_random)
    elif cfg.data.name == "miotcd" and balanced:
        label_balance = cfg.data.num_classes * 5
        datamodule.train_set.label_balanced(
            n_per_class=label_balance // num_classes, num_classes=num_classes
        )
        label_random = num_labelled - label_balance
        if label_random > 0:
            datamodule.train_set.label_randomly(label_random)
    elif balanced:
        datamodule.train_set.label_balanced(
            n_per_class=num_labelled // num_classes, num_classes=num_classes
        )
    else:
        datamodule.train_set.label_randomly(num_labelled)

    logger.info("{} Test Samples".format(len(datamodule.test_set)))

    training_loop = ActiveTrainingLoop(cfg, datamodule=datamodule, base_dir=os.getcwd())
    training_loop.init_callbacks()
    if len(training_loop.callbacks) < 3:
        raise NotImplementedError
    training_loop.init_model()
    training_loop.model.load_only_state_dict(ckpt_path)
    # training_loop.model = training_loop.model.to("cuda:0")
    training_loop.init_trainer()
    training_loop.test()

    imb_metric_callback: ImbClassMetricCallback = training_loop.callbacks[
        monitor_callback
    ]
    conf_mat = to_numpy(imb_metric_callback.pred_conf["test"].compute())
    conf_mat = pd.DataFrame(data=conf_mat)
    conf_mat.to_csv(path / "test-conf_mat.csv")

    test_dict = dict()
    test_dict = imb_metric_callback.compute_pred_metrics(mode="test")
    for key, val in imb_metric_callback.auc_dict.items():
        if "test" in key:
            test_dict[key] = val.compute()

    for key, val in test_dict.items():
        test_dict[key] = to_numpy(val)

    return test_dict


# path = "/home/c817h/Documents/logs/activelearning/sweep/miotcd/basic_lab-55_resnet_ep-200_drop-0_lr-0.1_wd-0.005_opt-sgd_trafo-imagenet_train/2022-09-01_17-15-31-766614"
# path = Path(path)

# path = "/home/c817h/network/Cluster-Experiments/activelearning/miotcd/active-miotcd_high/basic-pretrained_model-resnet_drop-0.5_aug-imagenet_train_acq-bald_ep-80_freeze-False_smallhead-False/2022-09-14_08-11-34-962453"


def test_active_exp(path, force_override):
    path = Path(path)
    paths = []
    if (path / "stored.npz").is_file():
        logger.info("Stored Exists")
        run_on = (path / "test_metrics.csv").is_file()
        if run_on:
            csv_dict = pd.read_csv(path / "test_metrics.csv", index_col=0)
            data_dict = csv_dict.to_dict(orient="list")
            run_on = len(data_dict) == 0
        if run_on or force_override:
            logger.info("Evaluating Path {}".format(path))
            for loop_path in path.iterdir():
                if loop_path.is_dir() and "loop-" in loop_path.name:
                    paths.append(loop_path)
            paths.sort(key=lambda x: int(x.name.__str__().split("-")[1]))

            test_metrics = []
            for loop_path in paths:
                logger.info("Evaluating Loop {}".format(path))
                config_path = loop_path / "hparams.yaml"
                cfg = load_omega_conf(config_path)
                cfg.trainer.data_root = os.getenv("DATA_ROOT")
                cfg.model.load_pretrained = None
                test_metrics.append(main(cfg, loop_path))

            df = pd.DataFrame(test_metrics)
            df.to_csv(path / "test_metrics.csv")
            return None
    logger.info("Skipping Path {}".format(path))
    return None


if __name__ == "__main__":
    force_override = False

    # path = Path(path)
    # config_path = path / "hparams.yaml"
    # print(config_path)
    # cfg = load_omega_conf(config_path)
    # main(cfg, path)
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-p", "--path", type=str)
    parser.add_argument("--glob", action="store_true")
    args = parser.parse_args()

    if args.glob:
        paths = Path(args.path)
        paths = [pth.parent for pth in paths.rglob("stored.npz")]

    else:
        paths = [Path(args.path)]

    for path in paths:
        test_active_exp(path, force_override)
