from argparse import ArgumentParser

from launcher import ExperimentLauncher

# Add Transformations from Randaugment and Changing of Learning Rates

config_dict = {
    "model": ["resnet"],
    "data": "cifar10_imb",
    # "optim": ["sgd"],
    "optim": ["sgd_cosine"],
}

hparam_dict = {
    "trainer.run_test": False,
    "model.weighted_loss": True,  # Weighted Loss is working correctly
    "active.num_labelled": [50, 250, 1000],
    "data.val_size": [50 * 5, 250 * 5, None],
    "model.dropout_p": [0],
    "model.learning_rate": [0.1, 0.01],  # is more stable than 0.1!
    "model.weight_decay": [5e-3, 5e-4],  # wd of 5e-2 does not help!
    # "model.use_ema": [True, False],
    "model.use_ema": False,
    "trainer.max_epochs": 200,
    "trainer.seed": [12345, 12346, 12347],
    "data.transform_train": ["cifar_randaugmentMC"],
    "trainer.precision": 16,
    "trainer.batch_size": 1024,
}

# naming_conv = (
#     "sweep_basic_{data}_lab-{active.num_labelled}_{model}_ep-{trainer.max_epochs}"
# )
naming_conv = "sweep/{data}/basic_lab-{active.num_labelled}_{model}_ep-{trainer.max_epochs}_drop-{model.dropout_p}_lr-{model.learning_rate}_wd-{model.weight_decay}_opt-{optim}_trafo-{data.transform_train}"

path_to_ex_file = "src/run_training.py"

joint_iteration = ["active.num_labelled", "data.val_size"]


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
