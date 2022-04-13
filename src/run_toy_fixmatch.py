import os
from models.fixmatch import FixMatch
from data.data import TorchVisionDM
import hydra
from omegaconf import DictConfig
from utils import config_utils

import utils

from run_toy import ToyActiveLearningLoop, get_toy_dm

# from trainer import ActiveTrainingLoop

active_dataset = True


@hydra.main(config_path="./config", config_name="config_toy_fixmatch")
def main(cfg: DictConfig):
    config_utils.print_config(cfg)
    train(cfg)


def train(cfg: DictConfig):
    utils.set_seed(cfg.trainer.seed)
    balanced = cfg.active.balanced
    num_classes = cfg.data.num_classes
    num_labelled = cfg.active.num_labelled

    datamodule = get_toy_dm(cfg, active_dataset)
    num_classes = cfg.data.num_classes
    if active_dataset:
        if balanced:
            datamodule.train_set.label_balanced(
                n_per_class=num_labelled // num_classes, num_classes=num_classes
            )
        else:
            datamodule.train_set.label_randomly(num_labelled)

    training_loop = FixToyTrainingLoop(
        cfg,
        datamodule,
        active=active_dataset,
        base_dir=os.getcwd(),
    )
    training_loop.main()
    if active_dataset:
        active_store = training_loop.active_callback()
    training_loop.log_save_dict()


class FixToyTrainingLoop(ToyActiveLearningLoop):
    def init_model(self):
        self.model = FixMatch(self.cfg)


if __name__ == "__main__":
    main()
