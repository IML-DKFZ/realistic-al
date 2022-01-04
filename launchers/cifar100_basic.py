from argparse import ArgumentParser
from launcher import BaseLauncher

config_dict = {
    "model": "resnet",
    "query": ["random", "entropy", "kcentergreedy", "bald"],
    "data": ["cifar10"],  # , "cifar100"],
    "active": ["standard"],  # did not run! "standard_250", "cifar10_low_data"
    "optim": ["sgd"],
}

hparam_dict = {
    "trainer.seed": [12345, 12346, 12347],
    "model.dropout_p": [0, 0.5],
    "model.learning_rate": [0.01],
    "model.use_ema": False,
}
naming_conv = "{data}_{active}_{query}_drop-{model.dropout_p}_goodfit"


joint_iteration = None

path_to_ex_file = "src/main.py"

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
