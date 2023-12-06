from argparse import ArgumentParser

from launcher import ExperimentLauncher

# Add Transformations from Randaugment and Changing of Learning Rates

config_dict = {
    "model": ["resnet_fixmatch"],  # , "wideresnet-cifar10"],
    # "model": "wideresnet-cifar10",
    "data": "cifar10",
    "active": "standard",
    "optim": "sgd_fixmatch",
}

hparam_dict = {
    "active.num_labelled": [40, 250, 4000],  # , 1000, 5000],
    "model.dropout_p": [0],
    "model.learning_rate": [0.003],  # is more stable than 0.1!
    "model.small_head": [False],
    "model.use_ema": [False],
    "model.weight_decay": [1e-4, 5e-5],
    "trainer.max_epochs": 50,
    "trainer.seed": [12345, 12346, 12347],
    "data.transform_train": [
        "cifar_basic",
        # "cifar_randaugment",
    ],
    "sem_sl.eman": [False],
    "model.load_pretrained": True,
}

naming_conv = (
    "sweep/{data}/fixmatch_basic_pretrained_lab-{active.num_labelled}_{model}_ep-{trainer.max_epochs}_smallhead-{model.small_head}_lr-{model.learning_rate}_wd-{model.weight_decay}"
    # "sweep_fixmatch_{data}_lab-{active.num_labelled}_{model}_ep-{trainer.max_epochs}"
)
# naming_conv = "sweep_fixmatch_{data}_{model}_{trainer.max_epochs}_{active.num_labelled}"  # {model}"
path_to_ex_file = "src/run_training_fixmatch.py"

joint_iteration = ["model.load_pretrained", "trainer.seed"]


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser = ExperimentLauncher.add_argparse_args(parser)
    # parser.add_argument()
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
