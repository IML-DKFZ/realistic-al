from argparse import ArgumentParser

from launcher import ExperimentLauncher

# Add Transformations from Randaugment and Changing of Learning Rates

config_dict = {
    "model": "resnet_fixmatch",
    "data": "cifar100",
    "active": "standard",
    "optim": "sgd_fixmatch",
}

hparam_dict = {
    "active.num_labelled": [500, 1000],  # , 1000, 5000],
    "data.val_size": [2500, None],
    "model.dropout_p": [0],
    "model.learning_rate": [
        0.3,
        0.03,
    ],  # does not change according to paper, so taken from CIFAR-10
    "model.small_head": [True],
    "model.weight_decay": [5 - 3, 1e-3, 5e-4],
    "model.use_ema": [False],
    "model.distr_align": True,
    "trainer.max_epochs": 200,
    "trainer.seed": [12345, 12346, 12347],
    "data.transform_train": [
        "cifar_basic",
    ],
    "sem_sl.eman": [False],
    "trainer.precision": 16,
}

naming_conv = "sweep/{data}/fixmatch_lab-{active.num_labelled}_model-{model}_ep-{trainer.max_epochs}_wd-{model.weight_decay}_distr_align-{model.distr_align}"
# naming_conv = "sweep_fixmatch_{data}_{model}_{trainer.max_epochs}_{active.num_labelled}"  # {model}"
path_to_ex_file = "src/run_training_fixmatch.py"

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
