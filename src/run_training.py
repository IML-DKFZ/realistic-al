import os
from data.base_datamodule import BaseDataModule
from data.data import TorchVisionDM
import hydra
from omegaconf import DictConfig, OmegaConf
from utils import config_utils
from loguru import logger
from utils.log_utils import setup_logger


import utils
from trainer import ActiveTrainingLoop


@hydra.main(config_path="./config", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    setup_logger()
    logger.info("Start logging")
    config_utils.print_config(cfg)
    train(cfg)


def get_torchvision_dm(cfg: DictConfig, active_dataset: bool = True) -> TorchVisionDM:
    """Initialize TorchVisionDM from config.

    Args:
        config (DictConfig): Config obtained
        active_dataset (bool, optional): . Defaults to True.

    Returns:
        TorchVisionDM: _description_
    """
    imbalance = None
    if "imbalance" in cfg.data:
        imbalance = cfg.data.imbalance
    val_size = None
    if "val_size" in cfg.data:
        val_size = cfg.data.val_size

    datamodule = TorchVisionDM(
        data_root=cfg.trainer.data_root,
        batch_size=cfg.trainer.batch_size,
        dataset=cfg.data.name,
        min_train=cfg.active.min_train,
        val_split=cfg.data.val_split,
        random_split=cfg.active.random_split,
        num_classes=cfg.data.num_classes,
        mean=cfg.data.mean,
        std=cfg.data.std,
        transform_train=cfg.data.transform_train,
        transform_test=cfg.data.transform_test,
        shape=cfg.data.shape,
        num_workers=cfg.trainer.num_workers,
        seed=cfg.trainer.seed,
        active=active_dataset,
        persistent_workers=cfg.trainer.persistent_workers,
        imbalance=imbalance,
        timeout=cfg.trainer.timeout,
        val_size=val_size,
    )

    return datamodule


def label_active_dm(
    cfg: DictConfig,
    num_labelled: int,
    balanced: bool,
    datamodule: BaseDataModule,
    balanced_per_cls: int = 5,
):
    """Label the Dataset according to rules.
    Specific for imbalanced datasets (miotcd, isic2019, isic2016 & datamodule.imbalance=True)

    Args:
        cfg (DictConfig): _description_
        num_labelled (int): _description_
        balanced (bool): _description_
        datamodule (BaseDataModule): _description_
        balanced_per_cls (int, optional): _description_. Defaults to 5.
    """
    cfg.data.num_classes = cfg.data.num_classes
    if cfg.data.name in ["isic2019", "miotcd", "isic2016"] and balanced:
        label_balance = cfg.data.num_classes * balanced_per_cls
        datamodule.train_set.label_balanced(
            n_per_class=label_balance // cfg.data.num_classes,
            num_classes=cfg.data.num_classes,
        )
        label_random = num_labelled - label_balance
        if label_random > 0:
            datamodule.train_set.label_randomly(label_random)
    elif datamodule.imbalance and balanced:
        label_balance = cfg.data.num_classes * balanced_per_cls
        datamodule.train_set.label_balanced(
            n_per_class=label_balance // cfg.data.num_classes,
            num_classes=cfg.data.num_classes,
        )
        label_random = num_labelled - label_balance
        if label_random > 0:
            datamodule.train_set.label_randomly(label_random)
    elif balanced:
        datamodule.train_set.label_balanced(
            n_per_class=num_labelled // cfg.data.num_classes,
            num_classes=cfg.data.num_classes,
        )
    else:
        datamodule.train_set.label_randomly(num_labelled)


@logger.catch
def train(cfg: DictConfig):
    active_dataset = cfg.active.num_labelled is not None
    logger.info("Set seed")
    utils.set_seed(cfg.trainer.seed)

    datamodule = get_torchvision_dm(cfg, active_dataset)
    if cfg.active.num_labelled:
        label_active_dm(cfg, cfg.active.num_labelled, cfg.active.balanced, datamodule)

    training_loop = ActiveTrainingLoop(
        cfg, datamodule, active=False, base_dir=os.getcwd()
    )
    training_loop.main()
    training_loop.log_save_dict()


if __name__ == "__main__":
    main()
