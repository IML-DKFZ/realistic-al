from argparse import ArgumentParser
from launcher import ExperimentLauncher
from config_launcher import get_pretrained_arch

config_dict = {
    "model": "resnet",
    "query": [
        "random",
        "entropy",
        "kcentergreedy",
        "bald",
        # "variationratios",
        # "batchbald",
    ],
    "data": ["cifar100"],
    "active": [
        # "standard",
        "cifar100",
    ],  # did not run! "standard_250", "cifar10_low_data"
    "optim": ["sgd"],
}

hparam_dict = {
    "data.val_size": None,
    "trainer.seed": [12345, 12346, 12347],
    "trainer.max_epochs": 80,  # Think about this before commiting (or sweep!)
    "model.dropout_p": [0, 0, 0, 0.5],
    "model.learning_rate": [0.001],
    "model.freeze_encoder": [False],  # possibly add True
    # "model.finetune": [True],
    "model.use_ema": False,
    "model.load_pretrained": True,
    "data.transform_train": "cifar_randaugment",
    # experiment with big head and frozen encoder
    # "model.freeze_encoder": True,
    "model.small_head": [False],
    "trainer.precision": 32,
}

naming_conv = "{data}/active-{active}/basic-pretrained_model-{model}_drop-{model.dropout_p}_aug-{data.transform_train}_acq-{query}_ep-{trainer.max_epochs}_freeze-{model.freeze_encoder}_smallhead-{model.small_head}"


joint_iteration = [
    ["query", "model.dropout_p"],
    ["model.load_pretrained", "trainer.seed"],
    ["active", "data.val_size"],
]

path_to_ex_file = "src/main.py"

if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--data", type=str, default=config_dict["data"])
    parser.add_argument("--model", type=str, default=config_dict["model"])
    ExperimentLauncher.add_argparse_args(parser)
    launcher_args = parser.parse_args()

    config_dict["data"] = launcher_args.data
    config_dict["model"] = launcher_args.model

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
