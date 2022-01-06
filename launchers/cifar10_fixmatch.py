from argparse import ArgumentParser
from launcher import ExperimentLauncher

# Add Transformations from Randaugment and Changing of Learning Rates

config_dict = {
    "model": "wideresnet-cifar10",
    "data": "cifar10",
    "active": ["cifar10_low_data"],  # standard
    "query": ["random", "entropy", "kcentergreedy", "bald"],
    "optim": "sgd_fixmatch",
}

hparam_dict = {
    "model.dropout_p": [0, 0, 0, 0.5],
    "model.learning_rate": 0.03,  # is more stable than 0.1!
    "model.small_head": [True],
    "model.use_ema": [True],
    "model.finetune": [False],
    # "model.load_pretrained": None, # if this is set to None, weird errors appear!
    "trainer.max_epochs": 2000,
    "trainer.seed": [12345, 12346, 12347],
    "data.transform_train": [
        "cifar_basic",
        # "cifar_randaugment",
    ],
    "sem_sl.eman": [False],
}

naming_conv = (
    "active_fixmatch_{data}_set-{active}_{model}_acq-{query}_ep-{trainer.max_epochs}"
)
path_to_ex_file = "src/main_fixmatch.py"

joint_iteration = ["query", "model.dropout_p"]


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    ExperimentLauncher.add_argparse_args(parser)
    launcher_args = parser.parse_args()

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
