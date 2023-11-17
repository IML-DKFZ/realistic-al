import os
from pathlib import Path

import pandas as pd
from loguru import logger
from omegaconf import DictConfig

import utils
from data.data import TorchVisionDM
from models.callbacks.metrics_callback import ImbClassMetricCallback
from run_training import get_torchvision_dm, label_active_dm
from trainer import ActiveTrainingLoop
from utils import config_utils
from utils.io import load_omega_conf
from utils.log_utils import setup_logger
from utils.tensor import to_numpy


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

    logger.info("Instantiating Datamodule")
    datamodule = get_active_torchvision_dm(cfg)

    logger.info("{} Test Samples".format(len(datamodule.test_set)))

    training_loop = ActiveTrainingLoop(
        cfg, datamodule=datamodule, base_dir=os.getcwd(), loggers=False
    )
    # training_loop.init_callbacks()
    if len(training_loop.callbacks) < 3:
        raise NotImplementedError
    # training_loop.init_model()
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
    test_dict = imb_metric_callback.compute_pred_metrics(mode="test", class_wise=True)
    for key, val in imb_metric_callback.auc_dict.items():
        if "test" in key:
            test_dict[key] = val.compute()

    for key, val in test_dict.items():
        test_dict[key] = to_numpy(val)

    return test_dict


def get_active_torchvision_dm(cfg):
    datamodule = get_torchvision_dm(cfg)
    label_active_dm(cfg, cfg.active.num_labelled, cfg.active.balanced, datamodule)
    return datamodule


def test_active_exp(path, force_override):
    path = Path(path)
    paths = []
    if (path / "stored.npz").is_file():
        logger.info("Stored Exists")
        run_on = (path / "test_metrics.csv").is_file()
        if run_on:
            csv_dict = pd.read_csv(path / "test_metrics.csv", index_col=0)
            data_dict = csv_dict.to_dict(orient="list")
            # print(data_dict)
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
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-p", "--path", type=str)
    parser.add_argument("--glob", action="store_true")
    parser.add_argument("-l", "--list-only", action="store_true")
    parser.add_argument("-f", "--force-override", action="store_true")
    args = parser.parse_args()

    if args.glob:
        paths = Path(args.path)
        paths = [pth.parent for pth in paths.rglob("stored.npz")]

    else:
        paths = [Path(args.path)]

    for path in paths:
        if args.list_only:
            print(path)
        else:
            test_active_exp(path, args.force_override)
