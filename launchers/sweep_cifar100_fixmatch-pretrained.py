from argparse import ArgumentParser
from launcher import ExperimentLauncher

# Add Transformations from Randaugment and Changing of Learning Rates

config_dict = {
    # "model": ["resnet18", "wideresnet-cifar10"],
    "model": "resnet_fixmatch",  # wideresnet-cifar10
    "data": "cifar100",
    "active": "standard",
    "optim": "sgd_fixmatch",
}

hparam_dict = {
    "active.num_labelled": [400, 2500, 10000],
    "model.dropout_p": [0, 0.5],
    "model.learning_rate": 0.003,  # according to EMAN paper
    "model.small_head": [True, False],  # Based on SelfMatch
    "model.use_ema": [True, False],  # FixMAtch
    "model.finetune": [True, False],
    "model.freeze_encoder": [True, False],
    "model.load_pretrained": True,
    "trainer.max_epochs": 2000,
    "trainer.seed": [12345],  # , 12346, 12347],
    "data.transform_train": [
        "cifar_basic",
        "cifar_randaugment",
    ],
    "sem_sl.eman": [True, False],  # EMAN Paper
}

naming_conv = "sweep_fixmatch-pretrained_{data}_lab-{active.num_labelled}_{model}_ep-{trainer.max_epochs}"
path_to_ex_file = "src/run_training_fixmatch.py"

joint_iteration = ["model.load_pretrained", "trainer.seed"]


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

    if "model.load_pretrained" in hparam_dict:
        hparam_dict["model.load_pretrained"] = ExperimentLauncher.finalize_paths(
            hparam_dict["model.load_pretrained"], on_cluster=launcher_args.cluster
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
