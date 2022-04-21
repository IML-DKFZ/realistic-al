import hydra
from omegaconf import DictConfig
from utils import config_utils
from run_training_fixmatch import FixTrainingLoop
import utils
from loguru import logger

from main import active_loop, get_torchvision_dm


@hydra.main(config_path="./config", config_name="config_fixmatch")
def main(cfg: DictConfig):
    logger.add(__file__.split(".")[0] + ".log")
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
