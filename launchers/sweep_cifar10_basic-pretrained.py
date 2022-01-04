from argparse import ArgumentParser
from launcher import ExperimentLauncher

config_dict = {
    "model": [
        "resnet18"
    ],  # , "wideresnet-cifar10"], currently there are only pretrained models for resnet18 available!
    "data": "cifar10",
}

# Pretrained models from Baseline Pytorch Lightning Bolts - for final results, use own version
load_pretrained = [
    "SSL/SimCLR/cifar10/2021-11-11_16:20:56.103061/checkpoints/last.ckpt",
    # "SSL/SimCLR/cifar10/2021-11-15_10:29:02.475176/checkpoints/last.ckpt",
    # "SSL/SimCLR/cifar10/2021-11-15_10:29:02.500429/checkpoints/last.ckpt",
]

hparam_dict = {
    "active.num_labelled": [40, 500, 1000, 5000],
    "model.dropout_p": [0, 0.5],
    "model.learning_rate": [0.001],  # is more stable than 0.1!
    # "model.use_ema": [True, False], # obtain best model without EMA and then check on this setting for benefits!
    "model.use_ema": False,
    "model.finetune": [True, False],
    "model.freeze_encoder": [True, False],
    "model.load_pretrained": load_pretrained,
    "trainer.max_epochs": 200,
    "trainer.seed": [12345],  # , 12346, 12347],
    "data.transform_train": [
        "cifar_basic",
        "cifar_randaugment",
    ],
}

joint_iteration = ["model.load_pretrained", "trainer.seed"]

naming_conv = "sweep_baseline_{data}_{trainer.max_epochs}"  # {model}"
path_to_ex_file = "src/run_training.py"


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
