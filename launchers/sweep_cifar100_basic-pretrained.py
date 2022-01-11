from argparse import ArgumentParser
from launcher import ExperimentLauncher

config_dict = {
    "model": [
        "resnet"
    ],  # , "wideresnet-cifar10"], currently there are only pretrained models for resnet18 available!
    "data": "cifar100",
    "optim": ["sgd"],
}

hparam_dict = {
    "active.num_labelled": [400, 2500, 10000],  # according to FixMatch
    "model.dropout_p": [0, 0.5],
    "model.learning_rate": [0.001],  # is more stable than 0.1!
    # "model.use_ema": [True, False], # obtain best model without EMA and then check on this setting for benefits!
    "model.use_ema": False,
    "model.finetune": [True, False],
    "model.freeze_encoder": [True, False],
    "model.load_pretrained": True,
    "trainer.max_epochs": 200,
    "trainer.seed": [12345],  # , 12346, 12347],
    "data.transform_train": [
        "cifar_basic",
        "cifar_randaugment",
    ],
}

joint_iteration = ["model.load_pretrained", "trainer.seed"]

naming_conv = "sweep_basic-pretrained_{data}_lab-{active.num_labelled}_{model}_ep-{trainer.max_epochs}"

path_to_ex_file = "src/run_training.py"


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    # parser.add_argument("--data", type=str, default=config_dict["data"])
    parser.add_argument("--model", type=str, default=config_dict["model"])
    ExperimentLauncher.add_argparse_args(parser)
    launcher_args = parser.parse_args()

    # config_dict["data"] = launcher_args.data
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
