from argparse import ArgumentParser

from launcher import ExperimentLauncher

config_dict = {
    "model": "bayesian_mnist",
    "data": ["mnist", "fashion_mnist"],
    "active": ["mnist_batchbald", "mnist_batchbald_double", "mnist_batchbald_start"],
    "query": [
        "random",
        "bald",
        "entropy",
        "batchbald",
        "kcentergreedy",
        "variationratios",
    ],
}

hparam_dict = {
    # "model.dropout_p": [0, 0.5],
    # "model.learning_rate": 0.01,  # is more stable than 0.1!
    # "model.use_ema": [True, False],
    "model.use_ema": False,
    "trainer.max_epochs": 200,
    "trainer.seed": [12345, 12346, 12347],
    "data.transform_train": "basic",
}

naming_conv = "{data}/active_basic_set-{active}_model-{model}_query-{query}_ep-{trainer.max_epochs}"
path_to_ex_file = "src/main.py"

joint_iteration = None


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
