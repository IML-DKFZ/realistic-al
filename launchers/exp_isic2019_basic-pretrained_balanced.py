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
    "data": ["isic2019"],
    "active": [
        "isic19_low",
        "isic19_med",
        "isic19_high",
    ],
    "optim": ["sgd"],
}

load_pretrained = [
    "SSL/isic2019/isic_resnet18/2022-08-23_12-06-55-852190/checkpoints/last.ckpt",  # Seed 12345
    "SSL/isic2019/isic_resnet18/2022-08-25_10-19-38-902165/checkpoints/last.ckpt",  # Seed 12346
    "SSL/isic2019/isic_resnet18/2022-08-25_10-19-38-902489/checkpoints/last.ckpt",  # Seed 12347
]
hparam_dict = {
    "data.balanced_sampling": True,
    "data.val_size": [200, 1000, None],
    "trainer.seed": [12345, 12346, 12347],
    "trainer.max_epochs": 80,
    "model.dropout_p": [0, 0, 0, 0.5, 0],
    "model.learning_rate": [0.001, 0.001, 0.001],
    "model.weight_decay": [5e-3, 5e-3, 5e-3],
    "model.freeze_encoder": [False],  # possibly add True
    "model.use_ema": False,
    "model.load_pretrained": load_pretrained,
    "data.transform_train": ["isic_train", "isic_randaugment", "isic_randaugment"],
    "model.small_head": [False],
    "trainer.precision": 32,
    "trainer.deterministic": True,
}

naming_conv = "{data}/active-{active}/basic-pretrained_model-{model}_drop-{model.dropout_p}_aug-{data.transform_train}_acq-{query}_ep-{trainer.max_epochs}_freeze-{model.freeze_encoder}_smallhead-{model.small_head}_balancsamp-{data.balanced_sampling}"


joint_iteration = [
    ["model.load_pretrained", "trainer.seed"],
    [
        "active",
        "data.val_size",
        "model.learning_rate",
        "model.weight_decay",
        "data.transform_train",
    ],
    ["query", "model.dropout_p"],
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
