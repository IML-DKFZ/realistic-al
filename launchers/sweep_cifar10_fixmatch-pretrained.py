from argparse import ArgumentParser

from launcher import ExperimentLauncher

# Add Transformations from Randaugment and Changing of Learning Rates

config_dict = {
    # "model": ["resnet18", "wideresnet-cifar10"],
    "model": "resnet_fixmatch",  # wideresnet-cifar10
    "data": "cifar10",
    "active": "standard",
    "optim": "sgd_fixmatch",
}

# Pretrained models from Baseline Pytorch Lightning Bolts - for final results, use own version
load_pretrained = [
    "SSL/SimCLR/cifar10/2021-11-11_16:20:56.103061/checkpoints/last.ckpt",
    "SSL/SimCLR/cifar10/2021-11-15_10:29:02.475176/checkpoints/last.ckpt",
    "SSL/SimCLR/cifar10/2021-11-15_10:29:02.500429/checkpoints/last.ckpt",
]

hparam_dict = {
    # "active.num_labelled": [50, 500, 1000, 5000],  # , 1000, 5000],
    "active.num_labelled": [40, 250, 4000],
    "model.dropout_p": [0],  # , 0.5],
    "model.learning_rate": [0.003],  # according to EMAN paper
    "model.small_head": [True, False],  # Based on SelfMatch
    # "model.use_ema": [True, False],  # FixMAtch
    # "model.finetune": [True, False],
    # "model.freeze_encoder": [True, False],
    "model.load_pretrained": True,
    "trainer.max_epochs": [200, 100],  # try 50!
    "trainer.seed": [12345, 12346, 12347],
    "data.transform_train": [
        "cifar_basic",
    ],
    "sem_sl.eman": [False],  # EMAN Paper
}
# TODO: Change naming convention!
naming_conv = "sweep/{data}/fixmatch_pretrained_lab-{active.num_labelled}_{model}_drop-{model.dropout_p}_lr-{model.learning_rate}_smallhead-{model.small_head}_ep-{trainer.max_epochs}"
# naming_conv = "sweep_fixmatch-pretrained_{data}_lab-{active.num_labelled}_{model}_ep-{trainer.max_epochs}"
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

    launcher = ExperimentLauncher(
        config_dict,
        hparam_dict,
        launcher_args,
        naming_conv,
        path_to_ex_file,
        joint_iteration=joint_iteration,
    )

    launcher.launch_runs()
