import hydra
from omegaconf import DictConfig
from utils import config_utils
import utils
from loguru import logger

import pytorch_lightning as pl

from trainer_fix import FixTrainingLoop
from main import active_loop, get_torchvision_dm
from utils.log_utils import setup_logger


@hydra.main(config_path="./config", config_name="config_fixmatch", version_base="1.1")
def main(cfg: DictConfig):
    setup_logger()
    logger.info("Start logging")
    config_utils.print_config(cfg)
    logger.info("Set seed")
    utils.set_seed(cfg.trainer.seed)

    active_loop(
        cfg,
        FixTrainingLoop,
        get_torchvision_dm,
        cfg.active.num_labelled,
        cfg.active.balanced,
        cfg.active.acq_size,
        cfg.active.num_iter,
    )


if __name__ == "__main__":
    main()
