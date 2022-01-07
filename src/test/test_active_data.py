from typing import List
import pytest
import os

import numpy as np
import torch
from torch.utils.data import Dataset

############# Needed to execute as main ############
import sys

src_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(src_folder)

####################################################

from utils import set_seed
from data.data import TorchVisionDM

shape = (3, 128, 128)
bs = 64
num_labelled = 100
SEED = 12345

label = torch.cat([torch.ones(bs // 2), torch.zeros(bs // 2)])
test_batch = torch.randn(bs, *shape)

dm_params = {
    "data_root": os.getenv("DATA_ROOT"),
}


def setup_active_datamodule(seed):
    set_seed(seed)
    dm = TorchVisionDM(**dm_params, seed=seed)
    dm.train_set.label_randomly(num_labelled)
    return dm


class TestSet(Dataset):
    def __init__(self, length=5000):
        super().__init__()
        self.data = torch.randn(*shape)
        self.label = label
        self.len = length

    def __getitem__(self, i):
        batch = (self.data, self.label)
        return batch

    def __len__(self):
        return self.len


def test_active_dataset():
    """Test whether two datasets are identical for the same Seed"""
    dm_1 = setup_active_datamodule(SEED)

    dm_2 = setup_active_datamodule(SEED)
    # check whether first layer, the split of training set is identical
    assert dm_1.train_set._dataset.indices == dm_2.train_set._dataset.indices
    # check whether the same samples are labeled
    assert (dm_1.train_set.labelled == dm_2.train_set.labelled).all()

    ## check whether the validation sets are identical
    assert dm_1.val_set.indices == dm_2.val_set.indices


def test_active_dataset_seeds():
    """Test whether two datasets differ for different SEEDS"""
    seed = SEED
    dm_1 = setup_active_datamodule(seed)

    train_ident = []
    val_ident = []
    label_ident = []
    for i in range(1, 10):
        seed = SEED + i

        dm_2 = setup_active_datamodule(seed)

        # check whether first layer, the split of training set is identical
        train_ident.append(
            dm_1.train_set._dataset.indices == dm_2.train_set._dataset.indices
        )
        # check whether the same samples are labeled
        label_ident.append((dm_1.train_set.labelled == dm_2.train_set.labelled).all())

        ## check whether the validation sets are identical
        val_ident.append(dm_1.val_set.indices == dm_2.val_set.indices)

    def check_one_not_true(x: List[bool]):
        return (~np.array(x)).any()

    # verify training_indices differ
    assert check_one_not_true(train_ident)

    # verify validation_indices differ
    assert check_one_not_true(val_ident)

    # verify labelling not identical
    assert check_one_not_true(label_ident)


if __name__ == "__main__":
    test_active_dataset()
    test_active_dataset_seeds()
