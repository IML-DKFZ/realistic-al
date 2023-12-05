from argparse import ArgumentParser

from launcher import ExperimentLauncher

# Add Transformations from Randaugment and Changing of Learning Rates

config_dict = {
    "model": "resnet_fixmatch",
    "data": "cifar10",
    "active": ["cifar10_low", "cifar10_med"],
    "query": [
        "random",
        "entropy",
        "kcentergreedy",
        "badge",
    ],
    "optim": "sgd_fixmatch",
}

hparam_dict = {
    "data.val_size": [250, 2500],
    "model.dropout_p": [0],  # , 0.5],
    "model.learning_rate": 0.03,  # is more stable than 0.1!
    "model.small_head": [True],
    "model.use_ema": [False],
    "model.finetune": [False],
    "model.load_pretrained": None,  # if this is set to None, weird errors appear!
    "trainer.max_epochs": 200,
    "trainer.seed": [12345, 12346, 12347],
    "data.transform_train": [
        "cifar_basic",
    ],
    "sem_sl.eman": [False],
    "trainer.precision": 32,
    "trainer.deterministic": True,
}

# naming_conv = (
#     "active_fixmatch_{data}_set-{active}_{model}_acq-{query}_ep-{trainer.max_epochs}"
# )

naming_conv = "{data}/active-{active}/fixmatch_model-{model}_drop-{model.dropout_p}_aug-{data.transform_train}_acq-{query}_ep-{trainer.max_epochs}"

path_to_ex_file = "src/main_fixmatch.py"

joint_iteration = ["data.val_size", "active"]


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
