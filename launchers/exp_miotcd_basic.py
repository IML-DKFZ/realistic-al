from argparse import ArgumentParser

from launcher import ExperimentLauncher

config_dict = {
    "model": ["resnet"],
    "data": "miotcd",
    "active": ["miotcd_low", "miotcd_med", "miotcd_high"],  # standard
    "query": ["random", "entropy", "kcentergreedy", "bald"],
    "optim": ["sgd_cosine"],
}


num_classes = 11
hparam_dict = {
    "trainer.run_test": True,
    "data.val_size": [num_classes * 5 * 5, num_classes * 25 * 5, num_classes * 100 * 5],
    "model.dropout_p": [0, 0, 0, 0.5],
    "model.learning_rate": [
        0.01,
        0.01,
        0.1,
    ],
    "model.weight_decay": [5e-3, 5e-4, 5e-3],
    "model.use_ema": False,
    "model.small_head": [True],
    "model.weighted_loss": True,
    "model.freeze_encoder": False,
    "trainer.max_epochs": 200,
    "trainer.batch_size": 512,
    "trainer.num_workers": 10,
    "trainer.seed": [12345, 12346, 12347],
    "data.transform_train": ["imagenet_randaugMC"],
    "trainer.precision": 16,
    "trainer.deterministic": True,
}

joint_iteration = [
    ["query", "model.dropout_p"],
    ["active", "model.weight_decay", "model.learning_rate", "data.val_size"],
]

naming_conv = "{data}/active-{active}/basic_model-{model}_drop-{model.dropout_p}_aug-{data.transform_train}_acq-{query}_ep-{trainer.max_epochs}_smallhead-{model.small_head}"

path_to_ex_file = "src/main.py"


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
