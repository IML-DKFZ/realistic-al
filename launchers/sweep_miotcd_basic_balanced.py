from argparse import ArgumentParser

from launcher import ExperimentLauncher

config_dict = {
    "model": ["resnet"],
    "data": "miotcd",
    "optim": ["sgd_cosine"],
}

num_classes = 11
hparam_dict = {
    "trainer.run_test": False,
    "active.num_labelled": [num_classes * 5, num_classes * 25, num_classes * 100],
    "data.val_size": [num_classes * 5 * 5, num_classes * 25 * 5, num_classes * 100 * 5],
    # "active.num_labelled": [200, 800],
    # "data.val_size": [1000, None],
    "model.dropout_p": [0],
    "model.learning_rate": [0.1, 0.01],
    "model.weight_decay": [5e-3, 5e-4],
    "model.use_ema": False,
    "model.small_head": [True],
    # "model.weighted_loss": True,
    "data.balanced_sampling": True,
    "trainer.max_epochs": 200,
    "trainer.batch_size": 512,
    "trainer.seed": [12345, 12346, 12347],
    "trainer.precision": 16,
    # "data.transform_train": ["imagenet_train", "imagenet_randaug",],
    # "data.transform_train": ["imagenet_randaug", "imagenet_randaug_cutout"],
    # "data.transform_train": ["imagenet_randaug_cutout"],
    "data.transform_train": [
        "imagenet_randaugMC",
        "imagenet_train",
        # "imagenet_randaug",
    ],
}

joint_iteration = [["active.num_labelled", "data.val_size"]]

naming_conv = "sweep/{data}/basic_lab-{active.num_labelled}_{model}_ep-{trainer.max_epochs}_drop-{model.dropout_p}_lr-{model.learning_rate}_wd-{model.weight_decay}_opt-{optim}_trafo-{data.transform_train}_balancsamp-{data.balanced_sampling}"

path_to_ex_file = "src/run_training.py"


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    ExperimentLauncher.add_argparse_args(parser)
    launcher_args = parser.parse_args()

    config_dict, hparam_dict = ExperimentLauncher.modify_params_for_args(
        launcher_args, config_dict, hparam_dict
    )

    launcher = ExperimentLauncher(
        config_dict,
        hparam_dict,
        launcher_args,
        naming_conv,
        path_to_ex_file,
        joint_iteration=joint_iteration,
    )

    launcher.launch_runs()
