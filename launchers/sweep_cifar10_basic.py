from argparse import ArgumentParser
from launcher import ExperimentLauncher

# Add Transformations from Randaugment and Changing of Learning Rates

config_dict = {
    "model": ["resnet", "wideresnet-cifar10"],
    "data": "cifar10",
}

hparam_dict = {
    "active.num_labelled": [40, 500, 1000, 5000],
    "model.dropout_p": [0, 0.5],
    "model.learning_rate": 0.01,  # is more stable than 0.1!
    # "model.use_ema": [True, False],
    "model.use_ema": False,
    "trainer.max_epochs": 200,
    "trainer.seed": [12345],  # , 12346, 12347],
    "data.transform_train": [
        "cifar_basic",
        "cifar_randaugment",
    ],
}

naming_conv = (
    "sweep_basic_{data}_lab-{active.num_labelled}_{model}_ep-{trainer.max_epochs}"
)
path_to_ex_file = "src/run_training.py"


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
    )

    launcher.launch_runs()
