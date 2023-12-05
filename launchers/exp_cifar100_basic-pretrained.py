from argparse import ArgumentParser

from config_launcher import get_pretrained_arch
from launcher import ExperimentLauncher

config_dict = {
    "model": "resnet",
    "query": [
        "random",
        "entropy",
        "kcentergreedy",
        "bald",
        "badge",
    ],
    "data": ["cifar100"],
    "active": [
        "cifar100_low",
        "cifar100_med",
        "cifar100_high",
    ],
    "optim": ["sgd"],
}

hparam_dict = {
    "data.val_size": [2500, None, None],  # None,
    "trainer.seed": [12345, 12346, 12347],
    "trainer.max_epochs": 80,
    "model.dropout_p": [0, 0, 0, 0.5, 0],
    "model.learning_rate": [0.001],
    "model.weight_decay": 5e-3,
    "model.freeze_encoder": [False],
    "model.use_ema": False,
    "model.load_pretrained": True,
    "data.transform_train": "cifar_randaugment",
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

    # This is only required for BADGE
    if launcher_args.bsub:
        launcher.ex_call = "~/run_active_20gb.sh python"

    launcher.launch_runs()
