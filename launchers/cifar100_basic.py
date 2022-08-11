from argparse import ArgumentParser
from launcher import ExperimentLauncher

config_dict = {
    "model": "resnet",
    "query": ["random", "entropy", "kcentergreedy", "bald"],  # , "variationratios"],
    "data": ["cifar100"],  # , "cifar100"],
    "active": [
        "cifar100_low",
        "cifar100_med",
        "cifar100_high",
        # "standard",
        # "cifar100",
    ],  # did not run! "standard_250", "cifar10_low_data"
    "optim": ["sgd"],
}

hparam_dict = {
    "data.val_size": [2500, None, None],  # None,
    "trainer.seed": [12345, 12346, 12347],
    "trainer.max_epochs": 200,
    "model.dropout_p": [0, 0, 0, 0.5],
    "model.learning_rate": [0.1],
    "model.load_pretrained": None,
    "model.use_ema": False,
    "data.transform_train": ["cifar_randaugment",],
    "trainer.precision": 32,
}

naming_conv = "{data}/active-{active}/basic_model-{model}_drop-{model.dropout_p}_aug-{data.transform_train}_acq-{query}_ep-{trainer.max_epochs}"


joint_iteration = [
    ["query", "model.dropout_p"],
    ["active", "data.val_size"],
]

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
