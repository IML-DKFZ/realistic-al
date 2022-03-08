import hydra
from omegaconf import DictConfig
from utils import config_utils
from run_toy import ToyActiveLearningLoop
import utils

from main import active_loop
from run_toy import get_toy_dm


@hydra.main(config_path="./config", config_name="config_toy")
def main(cfg: DictConfig):
    config_utils.print_config(cfg)
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


if __name__ == "__main__":
    main()
