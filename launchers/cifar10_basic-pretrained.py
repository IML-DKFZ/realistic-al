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
        "variationratios",
        # "batchbald",
    ],
    "data": ["cifar10"],  # , "cifar100"],
    "active": [
        "standard",
        "standard_250",
        "cifar10_low_data",
    ],  # did not run! "standard_250", "cifar10_low_data"
    "optim": ["sgd"],
}

# Pretrained models from Baseline Pytorch Lightning Bolts - for final results, use own version
load_pretrained = [
    "SSL/SimCLR/cifar10/2021-11-11_16:20:56.103061/checkpoints/last.ckpt",
    "SSL/SimCLR/cifar10/2021-11-15_10:29:02.475176/checkpoints/last.ckpt",
    "SSL/SimCLR/cifar10/2021-11-15_10:29:02.500429/checkpoints/last.ckpt",
]
hparam_dict = {
    "trainer.seed": [12345, 12346, 12347],
    "trainer.max_epochs": 40,  # Think about this before commiting (or sweep!)
    "model.dropout_p": [0, 0.5],
    "model.learning_rate": [0.001],
    "model.freeze_encoder": [True, False],
    "model.finetune": [False],
    "model.use_ema": False,
    "model.load_pretrained": True,
}

naming_conv = (
    "{data}/active-{active}/basic-pretrained_model-{model}_drop-{model.dropout_p}_aug-{data.transform_train}_acq-{query}_ep-{trainer.max_epochs}_freeze-{model.freeze_encoder}",
)


joint_iteration = ["model.load_pretrained", "trainer.seed"]

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
