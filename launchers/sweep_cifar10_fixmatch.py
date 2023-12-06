from argparse import ArgumentParser

from launcher import ExperimentLauncher

# Add Transformations from Randaugment and Changing of Learning Rates

config_dict = {
    "model": ["resnet_fixmatch"],
    # "model": "wideresnet-cifar10",
    "data": "cifar10",
    "active": "standard",
    "optim": "sgd_fixmatch",
}

hparam_dict = {
    "active.num_labelled": [50, 250],  # , 1000, 5000], # values from paper.
    "model.dropout_p": [0],
    "model.learning_rate": 0.03,  # according to paper!
    "model.small_head": [True],
    "model.use_ema": [False],
    "model.weight_decay": [5e-4],
    "trainer.max_epochs": 200,
    "trainer.seed": [12345, 12346, 12347],
    "data.transform_train": [
        "cifar_basic",
    ],
    "sem_sl.eman": [False],
    "model.load_pretrained": None,
    "trainer.precision": 16,
}

naming_conv = (
    "fixmatch_lab-{active.num_labelled}_model-{model}_ep-{trainer.max_epochs}_wd-{model.weight_decay}_lr-{model.learning_rate}"
    # "sweep_fixmatch_{data}_lab-{active.num_labelled}_{model}_ep-{trainer.max_epochs}"
)
# naming_conv = "sweep_fixmatch_{data}_{model}_{trainer.max_epochs}_{active.num_labelled}"  # {model}"
path_to_ex_file = "src/run_training_fixmatch.py"

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
