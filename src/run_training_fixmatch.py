import pytorch_lightning as pl
from torch.functional import norm
from torch.utils import data
from models.fixmatch import FixMatch
from data.data import TorchVisionDM
import hydra
from omegaconf import DictConfig
from utils import config_utils

import utils

from run_training import TrainingLoop

active_dataset = True


@hydra.main(config_path="./config", config_name="config_fixmatch")
def main(cfg: DictConfig):
    config_utils.print_config(cfg)
    train(cfg)


def train(cfg: DictConfig):
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
    )
    num_classes = cfg.data.num_classes
    if active_dataset:
        if balanced:
            datamodule.train_set.label_balanced(
                n_per_class=num_labelled // num_classes, num_classes=num_classes
            )
        else:
            datamodule.train_set.label_randomly(num_labelled)

    training_loop = FixTrainingLoop(cfg, datamodule, active=False)
    training_loop.main()


class FixTrainingLoop(TrainingLoop):
    def init_model(self):
        self.model = FixMatch(self.cfg)


if __name__ == "__main__":
    main()
