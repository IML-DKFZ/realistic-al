import os
import numpy as np

from data.skin_dataset import ISIC2019
from tqdm import tqdm


dataset = {
    "train": ISIC2019(
        root=os.getenv("DATA_ROOT"), train=True, preprocess=False, download=True
    ),
    "test": ISIC2019(
        root=os.getenv("DATA_ROOT"), train=False, preprocess=False, download=True
    ),
}
for data in dataset:
    dataset[data].train_test_split(force_override=True)
dataset_prep = ISIC2019(root=os.getenv("DATA_ROOT"), train=True, preprocess=True)

if __name__ == "__main__":
    # print(len(dataset))
    # for x, y in tqdm(dataset):
    #     pass

    # for x, y in tqdm(dataset_prep):
    #     pass

    for data in dataset:
        print("{} split has {} observations.".format(data, len(dataset[data])))
        targets, counts = np.unique(dataset[data].targets, return_counts=True)
        for target, count in zip(targets, counts):
            print("Target: {} -- Counts: {}".format(target, count))

    val_split = int(0.2 * len(dataset["train"].targets))
    datafinal = {
        "val": ISIC2019(
            root=os.getenv("DATA_ROOT"), train=True, preprocess=False, download=True
        ),
        "train_final": ISIC2019(
            root=os.getenv("DATA_ROOT"), train=True, preprocess=False, download=True
        ),
    }
    val_indices = np.random.choice(
        np.arange(len(datafinal["val"])), size=val_split, replace=False
    )
    train_mask = np.ones(shape=len(datafinal["train_final"]), dtype=np.bool8)
    train_mask[val_indices] = 0

    datafinal["train_final"].targets = datafinal["train_final"].targets[train_mask]
    datafinal["train_final"].data = [
        datafinal["train_final"].data[index]
        for index in range(len(datafinal["train_final"]))
        if train_mask[index]
    ]

    datafinal["val"].targets = datafinal["val"].targets[val_indices]
    datafinal["val"].data = [
        datafinal["val"].data[val_index] for val_index in val_indices
    ]
    for data in datafinal:
        print("{} split has {} observations.".format(data, len(datafinal[data])))
        targets, counts = np.unique(datafinal[data].targets, return_counts=True)
        for target, count in zip(targets, counts):
            print("Target: {} -- Counts: {}".format(target, count))

