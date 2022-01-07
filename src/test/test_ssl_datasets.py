from argparse import Namespace, ArgumentParser
import os

import numpy as np
import PIL
from numpy.lib.financial import ipmt

from torch.utils.data import DataLoader, ConcatDataset, SubsetRandomSampler
from torchvision import transforms

### Import Dataset
from torchvision import datasets
from data.datasets import (
    DTD,
    Flowers,
    Food,
    Aircraft,
    Caltech101,
    Cars,
    VOC2007,
    Pets,
    SUN397,
)

args = Namespace()
args.dataset = "cars"
# args.dataset = "caltech101"
args.batch_size = 128
args.image_size = 224
args.workers = 8
args.da = True
args.norm = True

# ISIC is missing here!
FINETUNE_DATASETS = {
    "aircraft": [Aircraft, "Aircraft", 100, "mean per-class accuracy"],
    "caltech101": [Caltech101, "Caltech101", 102, "mean per-class accuracy"],
    "cars": [Cars, "Cars", 196, "accuracy"],
    "cifar10": [datasets.CIFAR10, "CIFAR10", 10, "accuracy"],
    "cifar100": [datasets.CIFAR100, "CIFAR100", 100, "accuracy"],
    "dtd": [DTD, "DTD", 47, "accuracy"],
    "flowers": [Flowers, "Flowers", 102, "mean per-class accuracy"],
    "food": [Food, "Food", 101, "accuracy"],
    "pets": [Pets, "Pets", 37, "mean per-class accuracy"],
    "sun397": [SUN397, "SUN397", 397, "accuracy"],
    "voc2007": [VOC2007, "VOC2007", 20, "mAP"],
}


def get_dataset(dset, root, split, transform):
    # import pdb

    # pdb.set_trace()
    try:
        return dset(root, train=(split == "train"), transform=transform, download=True)
    except:
        return dset(root, split=split, transform=transform, download=True)


def get_train_valid_loader(
    dset,
    data_dir,
    normalise_dict,
    batch_size,
    image_size,
    random_seed,
    valid_size=0.2,
    shuffle=True,
    num_workers=1,
    pin_memory=True,
    data_augmentation=True,
):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset. A sample
    9x9 grid of the images can be optionally displayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - dset: dataset class to load
    - normalise_dict: dictionary containing the normalisation parameters of the training set
    - batch_size: how many samples per batch to load.
    - image_size: size of images after transforms
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert (valid_size >= 0) and (valid_size <= 1), error_msg

    normalize = transforms.Normalize(**normalise_dict)
    print("Train normaliser:", normalize)

    # define transforms with augmentations
    transform_aug = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, interpolation=PIL.Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    # define transform without augmentations
    transform_no_aug = transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )

    if not data_augmentation:
        transform_aug = transform_no_aug

    print("Train transform:", transform_aug)
    print("Val transform:", transform_no_aug)
    print("Trainval transform:", transform_aug)

    if dset in [Aircraft, DTD, Flowers, VOC2007]:
        # if we have a predefined validation set
        train_dataset = get_dataset(dset, data_dir, "train", transform_aug)
        valid_dataset_with_aug = get_dataset(dset, data_dir, "val", transform_aug)
        trainval_dataset = ConcatDataset([train_dataset, valid_dataset_with_aug])

        valid_dataset = get_dataset(dset, data_dir, "val", transform_no_aug)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        trainval_loader = DataLoader(
            trainval_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        # otherwise we select a random subset of the train set to form the validation set
        dataset = get_dataset(dset, data_dir, "train", transform_aug)
        valid_dataset = get_dataset(dset, data_dir, "train", transform_no_aug)

        num_train = len(dataset)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))

        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            sampler=valid_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        trainval_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return train_loader, valid_loader, trainval_loader


def get_test_loader(
    dset,
    data_dir,
    normalise_dict,
    batch_size,
    image_size,
    shuffle=False,
    num_workers=1,
    pin_memory=True,
):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR-10 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - dset: dataset class to load
    - normalise_dict: dictionary containing the normalisation parameters of the training set
    - batch_size: how many samples per batch to load.
    - image_size: size of images after transforms
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - data_loader: test set iterator.
    """

    normalize = transforms.Normalize(**normalise_dict)
    print("Test normaliser:", normalize)

    # define transform
    transform = transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )

    print("Test transform:", transform)

    dataset = get_dataset(dset, data_dir, "test", transform)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return data_loader


def prepare_data(
    dset,
    data_dir,
    batch_size,
    image_size,
    normalisation,
    num_workers,
    data_augmentation,
):
    print(
        f"Loading {dset} from {data_dir}, with batch size={batch_size}, image size={image_size}, norm={normalisation}"
    )
    if normalisation:
        normalise_dict = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    else:
        normalise_dict = {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]}
    train_loader, val_loader, trainval_loader = get_train_valid_loader(
        dset,
        data_dir,
        normalise_dict,
        batch_size,
        image_size,
        random_seed=0,
        num_workers=num_workers,
        pin_memory=False,
        data_augmentation=data_augmentation,
    )
    test_loader = get_test_loader(
        dset,
        data_dir,
        normalise_dict,
        batch_size,
        image_size,
        num_workers=num_workers,
        pin_memory=False,
    )
    return train_loader, val_loader, trainval_loader, test_loader


if __name__ == "__main__":
    from tqdm import tqdm

    dset, data_dir, num_classes, metric = FINETUNE_DATASETS[args.dataset]
    data_dir = os.path.join(os.getenv("DATA_ROOT"), data_dir)

    train_loader, val_loader, trainval_loader, test_loader = prepare_data(
        dset,
        data_dir,
        args.batch_size,
        args.image_size,
        normalisation=args.norm,
        num_workers=args.workers,
        data_augmentation=args.da,
    )

    def test_dataloader(dataloader):
        print("Number of samples in Dataset: {}".format(len(dataloader.dataset)))
        for i, batch in enumerate(tqdm(dataloader)):
            pass

    # test_dataloader(test_loader)
    test_dataloader(train_loader)
    test_dataloader(test_loader)

    # import IPython

    # IPython.embed()
