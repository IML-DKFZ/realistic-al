from argparse import ArgumentParser

from config_launcher import get_pretrained_arch
from launcher import ExperimentLauncher

config_dict = {
    "model": "resnet",
    "query": [
        "random",
        "entropy",
        "kcentergreedy",
        "badge",
        "bald",
        # "variationratios",
        "batchbald",
    ],
    "data": ["cifar10_imb"],
    "active": [
        "cifar10_low",
    ],
    "optim": ["sgd"],
}

# Pretrained models from Baseline Pytorch Lightning Bolts - for final results, use own version
load_pretrained = [
    "SSL/cifar10_imb/cifar_resnet18/2022-07-15_11-31-31-278397/checkpoints/last.ckpt",  # seed = 12345
    "SSL/cifar10_imb/cifar_resnet18/2022-07-15_11-31-34-631836/checkpoints/last.ckpt",  # seed = 12346
    "SSL/cifar10_imb/cifar_resnet18/2022-07-15_11-31-34-632366/checkpoints/last.ckpt",  # seed = 12347
]
hparam_dict = {
    "data.balanced_sampling": True,
    "data.val_size": [50 * 5],
    "trainer.seed": [12345, 12346, 12347],
    "trainer.max_epochs": 80,  # Think about this before commiting (or sweep!)
    "model.dropout_p": [0.5, 0.5, 0.5, 0, 0.5, 0.5],
    "model.learning_rate": [0.001],
    "model.freeze_encoder": [False],  # possibly add True
    "model.weight_decay": [5e-4],
    # "model.finetune": [True],
    "model.use_ema": False,
    "model.load_pretrained": load_pretrained,
    "data.transform_train": "cifar_randaugment",
    # experiment with big head and frozen encoder
    # "model.freeze_encoder": True,
    "model.small_head": [False],
    "trainer.precision": 32,
    "trainer.determinstic": True,
}

naming_conv = "{data}/active-{active}-batchbald/basic-pretrained_model-{model}_drop-{model.dropout_p}_aug-{data.transform_train}_acq-{query}_ep-{trainer.max_epochs}_freeze-{model.freeze_encoder}_smallhead-{model.small_head}_balancsamp-{data.balanced_sampling}"


joint_iteration = [
    ["model.load_pretrained", "trainer.seed"],
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
