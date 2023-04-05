import os
import numpy as np

from data.skin_dataset import ISIC2019
from tqdm import tqdm
from data.mio_dataset import MIOTCDDataset
from run_training import get_torchvision_dm


# dataset = {
#     "train": ISIC2019(
#         root=os.getenv("DATA_ROOT"), train=True, preprocess=False, download=True
#     ),
#     "test": ISIC2019(
#         root=os.getenv("DATA_ROOT"), train=False, preprocess=False, download=True
#     ),
# }
# for data in dataset:
#     dataset[data].train_test_split(force_override=True)
# dataset_prep = ISIC2019(root=os.getenv("DATA_ROOT"), train=True, preprocess=True)

# if __name__ == "__main__":
#     # print(len(dataset))
#     # for x, y in tqdm(dataset):
#     #     pass

#     # for x, y in tqdm(dataset_prep):
#     #     pass

#     for data in dataset:
#         print("{} split has {} observations.".format(data, len(dataset[data])))
#         targets, counts = np.unique(dataset[data].targets, return_counts=True)
#         for target, count in zip(targets, counts):
#             print("Target: {} -- Counts: {}".format(target, count))

#     val_split = int(0.2 * len(dataset["train"].targets))
#     datafinal = {
#         "val": ISIC2019(
#             root=os.getenv("DATA_ROOT"), train=True, preprocess=False, download=True
#         ),
#         "train_final": ISIC2019(
#             root=os.getenv("DATA_ROOT"), train=True, preprocess=False, download=True
#         ),
#     }
#     val_indices = np.random.choice(
#         np.arange(len(datafinal["val"])), size=val_split, replace=False
#     )
#     train_mask = np.ones(shape=len(datafinal["train_final"]), dtype=np.bool8)
#     train_mask[val_indices] = 0

#     datafinal["train_final"].targets = datafinal["train_final"].targets[train_mask]
#     datafinal["train_final"].data = [
#         datafinal["train_final"].data[index]
#         for index in range(len(datafinal["train_final"]))
#         if train_mask[index]
#     ]

#     datafinal["val"].targets = datafinal["val"].targets[val_indices]
#     datafinal["val"].data = [
#         datafinal["val"].data[val_index] for val_index in val_indices
#     ]
#     for data in datafinal:
#         print("{} split has {} observations.".format(data, len(datafinal[data])))
#         targets, counts = np.unique(datafinal[data].targets, return_counts=True)
#         for target, count in zip(targets, counts):
#             print("Target: {} -- Counts: {}".format(target, count))

if __name__ == "__main__":
    datasets = {
        "train": MIOTCDDataset(root=os.getenv("DATA_ROOT"), download=True),
        "test": MIOTCDDataset(root=os.getenv("DATA_ROOT"), download=True, train=False),
    }

    for data in datasets:
        print("{} split has {} observations.".format(data, len(datasets[data])))
        targets, counts = np.unique(datasets[data].targets, return_counts=True)
        for target, count in zip(targets, counts):
            print("Target: {} -- Counts: {}".format(target, count))

    # for data in datasets:
    #     print(data)
    #     # for i in tqdm(range(5000)):
    #     #     datasets[data][i]
    #     for _ in tqdm(datasets[data]):
    #         pass

# import math
# import os
# from typing import Callable, Union

# import hydra
# import numpy as np
# from omegaconf import DictConfig

# import utils
# from data.data import TorchVisionDM
# from trainer import ActiveTrainingLoop
# from run_training import get_torchvision_dm
# from utils import config_utils
# import time
# from loguru import logger
# from utils.log_utils import setup_logger

# import pandas as pd


# @hydra.main(config_path="./config", config_name="config", version_base="1.1")
# def main(cfg: DictConfig):
#     for data in ["isic2019", "cifar10_imb", "cifar10", "miotcd", "cifar100"]:
#         for seeds in [12345, 12346, 12347]:
#             setup_logger()
#             logger.info("Start logging")
#             config_utils.print_config(cfg)
#             logger.info("Set seed")
#             utils.set_seed(cfg.trainer.seed)

#             datamodule = get_torchvision_dm(cfg)
#             num_labelled = cfg.active.num_labelled
#             balanced = cfg.active.balanced
#             num_classes = cfg.data.num_classes
#             if cfg.data.name == "isic2019" and balanced:
#                 label_balance = 40
#                 datamodule.train_set.label_balanced(
#                     n_per_class=label_balance // num_classes, num_classes=num_classes
#                 )
#                 label_random = num_labelled - label_balance
#                 if label_random > 0:
#                     datamodule.train_set.label_randomly(label_random)
#             elif datamodule.imbalance and balanced:
#                 label_balance = cfg.data.num_classes * 5
#                 datamodule.train_set.label_balanced(
#                     n_per_class=label_balance // num_classes, num_classes=num_classes
#                 )
#                 label_random = num_labelled - label_balance
#                 if label_random > 0:
#                     datamodule.train_set.label_randomly(label_random)
#             elif cfg.data.name == "miotcd" and balanced:
#                 label_balance = cfg.data.num_classes * 5
#                 datamodule.train_set.label_balanced(
#                     n_per_class=label_balance // num_classes, num_classes=num_classes
#                 )
#                 label_random = num_labelled - label_balance
#                 if label_random > 0:
#                     datamodule.train_set.label_randomly(label_random)
#             elif balanced:
#                 datamodule.train_set.label_balanced(
#                     n_per_class=num_labelled // num_classes, num_classes=num_classes
#                 )
#             else:
#                 datamodule.train_set.label_randomly(num_labelled)

#             labels = []
#             for _, y in datamodule.train_set:
#                 labels.append(y)
#             labels = np.array(labels)
#             from pathlib import Path

#             imbalance = False
#             if hasattr(cfg.data, "imbalance"):
#                 imbalance = cfg.data.imbalance

#             dataname = cfg.data.name
#             if imbalance:
#                 dataname += "_imb"

#             save_folder = Path(cfg.save_path)
#             print(save_folder)
#             if save_folder.is_dir():
#                 print(f"Folder {save_folder} exists")
#             np.savez(
#                 save_folder
#                 / f"Data={dataname}-Setting={cfg.active.num_labelled}-Seed={cfg.trainer.seed}",
#                 initial_labels=labels,
#             )


# if __name__ == "__main__":
#     main()

# """
# python play.py --multirun data=cifar10,cifar10_imb trainer.seed=12345,12346,12347 active=cifar10_low,cifar10_med,cifar10_high ++save_path=/home/c817h/Documents/projects/Active_Learning/activeframework/dump
# python play.py --multirun data=cifar10_imb trainer.seed=12345,12346,12347 active=cifar10_low,cifar10_med,cifar10_high ++save_path=/home/c817h/Documents/projects/Active_Learning/activeframework/dump
# python play.py --multirun data=isic2019 trainer.seed=12345,12346,12347 active=isic19_low,isic19_med,isic19_high ++save_path=/home/c817h/Documents/projects/Active_Learning/activeframework/dump
# python play.py --multirun data=cifar100 trainer.seed=12345,12346,12347 active=cifar100_low,cifar100_med,cifar100_high ++save_path=/home/c817h/Documents/projects/Active_Learning/activeframework/dump
# python play.py --multirun data=miotcd trainer.seed=12345,12346,12347 active=miotcd_low,miotcd_med,miotcd_high ++save_path=/home/c817h/Documents/projects/Active_Learning/activeframework/dump

# """
