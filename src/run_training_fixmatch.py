import os
from data.data import TorchVisionDM
import hydra
from omegaconf import DictConfig
from loguru import logger
from utils.log_utils import setup_logger
from utils import config_utils

import utils

from trainer_fix import FixTrainingLoop

active_dataset = True


@hydra.main(config_path="./config", config_name="config_fixmatch", version_base="1.1")
def main(cfg: DictConfig):
    setup_logger()
    logger.info("Start logging")
    config_utils.print_config(cfg)
    train(cfg)


@logger.catch
def train(cfg: DictConfig):
    logger.info("Set seed")
    utils.set_seed(cfg.trainer.seed)
    balanced = cfg.active.balanced
    num_classes = cfg.data.num_classes
    num_labelled = cfg.active.num_labelled

    datamodule = TorchVisionDM(
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
        if cfg.data.name == "isic2019" and balanced:
            label_balance = 40
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

    training_loop = FixTrainingLoop(cfg, datamodule, active=False, base_dir=os.getcwd())
    training_loop.main()
    training_loop.log_save_dict()


if __name__ == "__main__":
    main()
