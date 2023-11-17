from pathlib import Path

import hydra
import matplotlib.pyplot as plt
from loguru import logger
from omegaconf import DictConfig

import utils
from main import active_loop
from run_toy import ToyActiveLearningLoop, get_toy_dm
from toy_callback import ToyVisCallback
from utils import config_utils
from utils.file_utils import get_experiment_dicts
from utils.log_utils import setup_logger


@hydra.main(config_path="./config", config_name="config_toy", version_base="1.1")
def main(cfg: DictConfig):
    setup_logger()
    # logger.add(__file__.split(".")[0] + ".log")
    logger.info("Start logging")
    config_utils.print_config(cfg)
    logger.info("Set seed")
    utils.set_seed(cfg.trainer.seed)

    active_loop(
        cfg,
        ToyActiveLearningLoop,
        get_toy_dm,
        cfg.active.num_labelled,
        cfg.active.balanced,
        cfg.active.acq_size,
        cfg.active.num_iter,
    )

    experiment_path = Path(".").resolve()

    dictlist = get_experiment_dicts(experiment_path)
    fig, axs = ToyVisCallback.fig_full_vis_2d(dictlist)
    plt.savefig(experiment_path / "fig_full_vis.png")


if __name__ == "__main__":
    main()
