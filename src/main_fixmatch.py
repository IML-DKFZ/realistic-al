import hydra
from omegaconf import DictConfig
from utils import config_utils
from run_training_fixmatch import FixTrainingLoop
import utils

from main import active_loop


@hydra.main(config_path="./config", config_name="config_fixmatch")
def main(cfg: DictConfig):
    config_utils.print_config(cfg)
    utils.set_seed(cfg.trainer.seed)

    active_loop(
        cfg,
        FixTrainingLoop,
        cfg.active.num_labelled,
        cfg.active.balanced,
        cfg.active.acq_size,
        cfg.active.num_iter,
    )


if __name__ == "__main__":
    main()
