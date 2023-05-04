from typing import List
from omegaconf import OmegaConf
import pytest
import os
from pathlib import Path
import shutil


import numpy as np
import torch

############# Needed to execute as main ############
import sys

src_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(src_folder)

####################################################

from utils.setup import set_seed
import utils.path_utils as path_utils
import utils.io as io
from run_toy import ToyActiveLearningLoop, get_toy_dm

# DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = path_utils.test_data_folder


@pytest.fixture
def data_dir():
    return DATA_DIR


# TODO: Write test that ensures that whatever happens in between -- the random data labelling stays identical!


# tmp_path is generally obtainable in pytest which shows a folder structure created by pytest
@pytest.fixture
def tmp_test_dir(tmp_path, data_dir):
    # HACK: The analysis script deduces the exp name from the path
    tmp_test_path: Path = tmp_path / "tests" / "tests"
    tmp_test_path.mkdir(parents=True, exist_ok=True)

    # shutil.copy(data_dir / "config.yaml", tmp_test_path)
    # shutil.copy(data_dir / "raw_output.npz", tmp_test_path)
    # shutil.copy(data_dir / "external_confids.npz", tmp_test_path)

    # if (data_dir / "raw_output_dist.npz").is_file():
    #     shutil.copy(data_dir / "raw_output_dist.npz", tmp_test_path)

    # if (data_dir / "external_confids_dist.npz").is_file():
    #     shutil.copy(data_dir / "external_confids_dist.npz", tmp_test_path)

    return tmp_test_path


# this sets variables for pytest intern os environments
@pytest.fixture
def mock_env(monkeypatch, tmp_test_dir):
    monkeypatch.setenv("EXPERIMENT_ROOT", str(tmp_test_dir))
    monkeypatch.setenv("DATA_ROOT", "")


def test_identical_queries_same_seed_same_model(tmp_test_dir, data_dir, mock_env):
    active_stores = []
    for i in range(2):
        cfg = OmegaConf.load(os.path.join(data_dir, "run_toy.yaml"))
        active_dataset = True
        set_seed(cfg.trainer.seed)
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
        base_dir = os.path.join(
            cfg.trainer.experiments_root,
            cfg.trainer.experiment_name,
            cfg.trainer.experiment_id,
        )
        training_loop = ToyActiveLearningLoop(
            cfg, datamodule, active=active_dataset, base_dir=base_dir
        )
        training_loop.main()

        if active_dataset:
            active_store = training_loop.active_callback()
        active_stores.append(active_store)
    requests1 = active_stores[0].requests
    requests2 = active_stores[1].requests
    assert np.all(requests1 == requests2)
