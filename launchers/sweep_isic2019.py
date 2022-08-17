from argparse import ArgumentParser
from launcher import ExperimentLauncher

config_dict = {
    "model": ["resnet"],
    "data": "cifar10",
    "optim": ["sgd"],
}

hparam_dict = {
    "active.num_labelled": [50, 500, 1000, 5000, 10000],
    "data.val_size": [250, 2500, None, None, None],
    "model.dropout_p": [0],
    "model.learning_rate": [0.1],
    # "model.weight_decay": [5e-4, 5e-2],
    "model.weight_decay": [5e-3, 5e-4],
    # "model.learning_rate": [0.1, 0.01],  # is more stable than 0.1!
    "model.use_ema": False,
    # "model.finetune": [True, False],
    # "model.freeze_encoder": [True, False],
    "model.small_head": [True],
    "trainer.max_epochs": 200,
    "trainer.seed": [12345, 12346, 12347],
    "data.transform_train": ["cifar_randaugment"],
}

joint_iteration = [["active.num_labelled", "data.val_size"]]

naming_conv = "sweep/{data}/basic_lab-{active.num_labelled}_{model}_ep-{trainer.max_epochs}_drop-{model.dropout_p}_lr-{model.learning_rate}_wd-{model.weight_decay}_opt-{optim}"
