from argparse import ArgumentParser
from launcher import ExperimentLauncher

config_dict = {
    "model": "resnet",
    "query": [
        "random",
        "entropy",
        "kcentergreedy",
        "bald",
        "variationratios",
        # "batchbald",
    ],
    "data": ["cifar10"],  # , "cifar100"],
    "active": [
        "standard",
        "standard_250",
        "cifar10_low_data",
    ],  # did not run! "standard_250", "cifar10_low_data"
    "optim": ["sgd"],
}

hparam_dict = {
    "trainer.seed": [12345, 12346, 12347],
    "trainer.max_epochs": 200,
    "model.dropout_p": [0, 0.5],
    "model.learning_rate": [0.01],
    "model.use_ema": False,
    "data.transform_train": [
        "cifar_basic",
        "cifar_randaugment",
    ],
}
naming_conv = (
    "{data}/active-{active}/model-{model}_drop-{model.dropout_p}_aug-{data.transform_train}_acq-{query}_ep-{trainer.max_epochs}"
    # "active_basic_{data}_set-{active}_{model}_acq-{query}_ep-{trainer.max_epochs}"
)

joint_iteration = None

path_to_ex_file = "src/main.py"

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
