from argparse import ArgumentParser

from config_launcher import get_pretrained_arch
from launcher import ExperimentLauncher

config_dict = {
    "model": "resnet",
    "query": [
        # "random",
        # "entropy",
        # "kcentergreedy",
        # "bald",
        "badge",
    ],
    # "data": ["cifar10"],  # , "cifar100"],
    "data": ["cifar10_imb"],  # , "cifar100"],
    # "active": ["cifar10_imb"],  # did not run! "standard_250", "cifar10_low_data"
    "active": [
        "cifar10_low",
        "cifar10_med",
        "cifar10_high",
    ],  # did not run! "standard_250", "cifar10_low_data"
    "optim": ["sgd"],
}

# Pretrained models from Baseline Pytorch Lightning Bolts - for final results, use own version
load_pretrained = [
    "SSL/cifar10_imb/cifar_resnet18/2022-07-15_11-31-31-278397/checkpoints/last.ckpt",  # seed = 12345
    "SSL/cifar10_imb/cifar_resnet18/2022-07-15_11-31-34-631836/checkpoints/last.ckpt",  # seed = 12346
    "SSL/cifar10_imb/cifar_resnet18/2022-07-15_11-31-34-632366/checkpoints/last.ckpt",  # seed = 12347
]
hparam_dict = {
    "data.val_size": [50 * 5, 250 * 5, None],
    "trainer.seed": [12345, 12346, 12347],
    "trainer.max_epochs": 80,  # Think about this before commiting (or sweep!)
    # "model.dropout_p": [0, 0, 0, 0.5, 0],
    "model.dropout_p": [0],
    "model.learning_rate": [0.001],
    "model.freeze_encoder": [False],  # possibly add True
    "model.weight_decay": [5e-3, 5e-3, 5e-4],
    "model.weighted_loss": True,
    # "model.finetune": [True],
    "model.use_ema": False,
    "model.load_pretrained": load_pretrained,
    "data.transform_train": "cifar_basic",
    # experiment with big head and frozen encoder
    # "model.freeze_encoder": True,
    "model.small_head": [False],
    "trainer.precision": 32,
    "trainer.deterministic": True,
}

naming_conv = "{data}/active-{active}/basic-pretrained_model-{model}_drop-{model.dropout_p}_aug-{data.transform_train}_acq-{query}_ep-{trainer.max_epochs}_freeze-{model.freeze_encoder}_smallhead-{model.small_head}__wloss-{model.weighted_loss}"


joint_iteration = ["model.load_pretrained", "trainer.seed"]

joint_iteration_2 = ["model.dropout_p", "query"]

joint_iteration_3 = ["data.val_size", "active", "model.weight_decay"]

joint_iteration = [joint_iteration, joint_iteration_2, joint_iteration_3]

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
