from argparse import ArgumentParser

from launcher import ExperimentLauncher

# Add Transformations from Randaugment and Changing of Learning Rates

config_dict = {
    "model": "resnet_fixmatch",  # wideresnet-cifar10
    "data": "cifar10",
    "active": [
        # "cifar10_low_data",
        # "standard_250",
        # "cifar10_low_data",
        "standard",
        # "cifar10_low_data",
    ],  # standard
    # "query": ["random", "entropy", "kcentergreedy", "bald"],
    "query": ["entropy"],
    "optim": "sgd_fixmatch",
}

# Pretrained models from Baseline Pytorch Lightning Bolts - for final results, use own version
load_pretrained = [
    "SSL/SimCLR/cifar10/2021-11-11_16:20:56.103061/checkpoints/last.ckpt",
    "SSL/SimCLR/cifar10/2021-11-15_10:29:02.475176/checkpoints/last.ckpt",
    "SSL/SimCLR/cifar10/2021-11-15_10:29:02.500429/checkpoints/last.ckpt",
]

hparam_dict = {
    "active.num_iter": 10,
    "model.dropout_p": [0.5],
    # "model.learning_rate": 0.003,  # is more stable than 0.1!
    "model.learning_rate": 0.01,
    "model.small_head": [False],
    # "model.use_ema": [True],
    "model.use_ema": False,
    # "model.finetune": [True],
    "model.finetune": False,
    "model.load_pretrained": True,
    "trainer.max_epochs": 200,
    "trainer.seed": [12345, 12346, 12347],
    "data.transform_train": [
        "cifar_basic",
        # "cifar_randaugment",
    ],
    # "sem_sl.eman": [True],
    "sem_sl.eman": [False],
    "model.freeze_encoder": True,
    # "model.freeze_encoder": False,
    "trainer.deterministic": True,
    "trainer.num_workers": 10,
    "trainer.persistent_workers": [
        # True,
        False,
    ],
}

# naming_conv = "active_fixmatch-pretrained_{data}_set-{active}_{model}_acq-{query}_ep-{trainer.max_epochs}"
# naming_conv = "{data}/active-{active}/fixmatch-pretrained_model-{model}_drop-{model.dropout_p}_aug-{data.transform_train}_acq-{query}_ep-{trainer.max_epochs}_freeze-{model.freeze_encoder}_smallhead-{model.small_head}"
# naming_conv = "{data}/debug/SEED_fixmatch-pretrained_model_seed-{trainer.seed}"
naming_conv = "{data}/debug/full-iter_fixmatch-pretrained_model_seed-{trainer.seed}_persistent-{trainer.persistent_workers}"

path_to_ex_file = "src/main_fixmatch.py"

joint_iteration = ["trainer.seed", "model.load_pretrained"]


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--data", type=str, default=config_dict["data"])
    parser.add_argument("--model", type=str, default=config_dict["model"])
    parser.add_argument("--active", type=str, default=config_dict["active"])
    ExperimentLauncher.add_argparse_args(parser)
    launcher_args = parser.parse_args()

    config_dict["data"] = launcher_args.data
    config_dict["model"] = launcher_args.model
    config_dict["active"] = launcher_args.active

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
