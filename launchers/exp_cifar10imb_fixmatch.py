from argparse import ArgumentParser

from launcher import ExperimentLauncher

config_dict = {
    "model": "resnet",
    "query": ["random", "entropy", "kcentergreedy", "badge"],
    "data": ["cifar10_imb"],
    "active": [
        "cifar10_low",
        "cifar10_med",
    ],
    "optim": ["sgd_fixmatch"],
}

hparam_dict = {
    "data.val_size": [50 * 5, 250 * 5],
    "active.acq_size": [150, 750],
    "active.num_iter": [4],
    "trainer.seed": [12345, 12346, 12347],
    "trainer.deterministic": True,
    "trainer.max_epochs": 200,
    # "model.dropout_p": [0, 0, 0, 0],
    "model.dropout_p": [0],
    "model.weight_decay": [1e-3, 1e-3],
    "model.weighted_loss": True,
    "model.learning_rate": [0.03],
    "model.small_head": [True],
    "model.use_ema": False,
    "data.transform_train": [
        "cifar_basic",
    ],
    "sem_sl.eman": [False],
    "model.load_pretrained": None,
    "model.distr_align": True,
    "trainer.timeout": 10,
}
naming_conv = "{data}/active-{active}/fixmatch_model-{model}_drop-{model.dropout_p}_aug-{data.transform_train}_acq-{query}_ep-{trainer.max_epochs}"

joint_iteration = [
    ["data.val_size", "active", "active.acq_size", "model.weight_decay"],
]

path_to_ex_file = "src/main_fixmatch.py"

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
