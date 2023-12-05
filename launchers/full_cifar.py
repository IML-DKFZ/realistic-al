from argparse import ArgumentParser

from launcher import ExperimentLauncher

config_dict = {
    "model": "resnet",
    "query": [
        "random",
    ],
    "data": ["cifar10", "cifar100", "cifar10_imb"],
    # "data": ["cifar10"],
    # "data": "cifar100",
    "active": [
        "full_data",
    ],
    "optim": ["sgd", "sgd_cosine"],
    # "optim": ["sgd"],
}

# fastest training: BS=1024, Prec=16
hparam_dict = {
    "trainer.seed": [12345, 12346, 12347],
    "trainer.max_epochs": 200,
    "model.dropout_p": [0],
    "model.learning_rate": [0.1],  # , 0.01],
    "model.weight_decay": [5e-3, 5e-4],  # [5e-3, 5e-4],
    "model.use_ema": False,
    "model.weighted_loss": [False, False, True],
    # "model.weighted_loss": [True],
    "trainer.batch_size": 1024,  # note: 128 and 256 make training much faster!
    # only to be continous with old experiments.
    # "data.transform_train": ["cifar_randaugment",],
    "data.transform_train": [
        "cifar_randaugmentMC",
    ],
    "trainer.precision": 16,
    "trainer.deterministic": True,
    "trainer.max_epochs": 200,
}
naming_conv = (
    "{data}/{active}/basic_model-{model}_drop-{model.dropout_p}_aug-{data.transform_train}_wd-{model.weight_decay}_lr-{model.learning_rate}_optim-{optim}"
    # "active_basic_{data}_set-{active}_{model}_acq-{query}_ep-{trainer.max_epochs}"
)

joint_iteration = None

path_to_ex_file = "src/run_training.py"

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
