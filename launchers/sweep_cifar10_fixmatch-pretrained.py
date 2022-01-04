from argparse import ArgumentParser
from launcher import BaseLauncher

from argparse import ArgumentParser
from launcher import BaseLauncher

# Add Transformations from Randaugment and Changing of Learning Rates

config_dict = {
    # "model": ["resnet18", "wideresnet-cifar10"],
    "model": "wideresnet-cifar10",
    "data": "cifar10",
    "active": "standard",
}

hparam_dict = {
    "active.num_labelled": [40],  # , 1000, 5000],
    "model.dropout_p": [0, 0.5],
    "model.learning_rate": 0.03,  # is more stable than 0.1!
    "model.small_head": [True, False],
    "model.use_ema": [True, False],
    "trainer.max_epochs": 2000,
    "trainer.seed": [12345],  # , 12346, 12347],
    "data.transform_train": [
        "cifar_basic",
        # "cifar_randaugment",
    ],
    "sem_sl.eman": [True, False],
}

naming_conv = "sweep_fixmatch-pretrained_{data}_{model}_{trainer.max_epochs}_{active.num_labelled}"
path_to_ex_file = "src/run_training.py"

joint_iteration = None


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    BaseLauncher.add_argparse_args(parser)
    launcher_args = parser.parse_args()

    config_dict, hparam_dict = BaseLauncher.modify_params_for_args(
        launcher_args, config_dict, hparam_dict
    )

    launcher = BaseLauncher(
        config_dict,
        hparam_dict,
        launcher_args,
        naming_conv,
        path_to_ex_file,
        joint_iteration=joint_iteration,
    )

    launcher.launch_runs()
