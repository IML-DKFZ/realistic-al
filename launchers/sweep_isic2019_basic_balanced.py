from argparse import ArgumentParser

from launcher import ExperimentLauncher

config_dict = {
    "model": ["resnet"],
    "data": "isic2019",
    # "optim": ["sgd"],
    "optim": "sgd_cosine",
}

hparam_dict = {
    "trainer.run_test": False,
    "active.num_labelled": [40, 200, 800],
    "data.val_size": [200, 1000, None],
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
    "trainer.seed": [12345, 12346, 12347],
    # "data.transform_train": ["isic_train", "isic_randaugment"],
    # "data.transform_train": "isic_randaugtensor",
    "data.transform_train": ["isic_train", "isic_randaugmentMC"],
    "trainer.deterministic": True,
    "trainer.batch_size": 512,
    "trainer.precision": 16,
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
