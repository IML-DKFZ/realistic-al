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
    "trainer.run_test": False,
    "active.num_labelled": [500, 1000],
    "data.val_size": [2500, None],
    "model.dropout_p": [0],
    "model.learning_rate": 0.03,  # according to EMAN paper
    "model.weight_decay": [1e-5, 1e-6, 0],
    "model.small_head": [False],  # Based on SelfMatch
    "model.use_ema": [False],  # FixMAtch
    "model.finetune": [False],
    "model.freeze_encoder": [False],
    "model.load_pretrained": True,
    "trainer.max_epochs": 200,
    "trainer.seed": [12345],  # , 12346, 12347],
    "data.transform_train": ["cifar_basic"],
    "sem_sl.eman": [False],  # EMAN Paper
}

naming_conv = "sweep/{data}/fixmatchv2_lab-pretrained_lab-{active.num_labelled}_model-{model}_ep-{trainer.max_epochs}_wd-{model.weight_decay}"
path_to_ex_file = "src/run_training_fixmatch.py"

joint_iteration = [
    ["model.load_pretrained", "trainer.seed"],
    ["active.num_labelled", "data.val_size"],
]


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
