import os

import hydra
from loguru import logger
from omegaconf import DictConfig

import utils
from data.data import TorchVisionDM
from run_training import get_torchvision_dm, label_active_dm
from trainer_fix import FixTrainingLoop
from utils import config_utils
from utils.log_utils import setup_logger

active_dataset = True


@hydra.main(config_path="./config", config_name="config_fixmatch", version_base="1.1")
def main(cfg: DictConfig):
    setup_logger()
    logger.info("Start logging")
    config_utils.print_config(cfg)
    train(cfg)


@logger.catch
def train(cfg: DictConfig):
    """Run standard training for FixMatch.

    Args:
        cfg (DictConfig): config from main
    """
    logger.info("Set seed")
    utils.set_seed(cfg.trainer.seed)
    datamodule = get_torchvision_dm(cfg, active_dataset=True)
    label_active_dm(cfg, cfg.active.num_labelled, cfg.active.balanced, datamodule)

    training_loop = FixTrainingLoop(cfg, datamodule, active=False, base_dir=os.getcwd())
    training_loop.main()
    training_loop.log_save_dict()


if __name__ == "__main__":
    main()
