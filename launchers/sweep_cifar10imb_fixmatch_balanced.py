from argparse import ArgumentParser

from launcher import ExperimentLauncher

# Add Transformations from Randaugment and Changing of Learning Rates

config_dict = {
    "model": ["resnet_fixmatch"],
    "data": "cifar10_imb",
    "active": "standard",
    "optim": "sgd_fixmatch",
}

hparam_dict = {
    "trainer.run_test": False,
    # "model.weighted_loss": [True],
    "data.balanced_sampling": True,
    "active.num_labelled": [50, 250],
    "data.val_size": [50 * 5, 250 * 5],
    "model.dropout_p": [0],
    "model.weight_decay": [
        1e-3,
        5e-4,
    ],  # 5e-3 was test beforehand leading to bad performance!
    "model.learning_rate": 0.03,  # is more stable than 0.1!
    "model.small_head": [True],
    "model.use_ema": [False],
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
    "sweep/{data}/fixmatch_lab-{active.num_labelled}_model-{model}_ep-{trainer.max_epochs}_wd-{model.weight_decay}_wloss-{model.weighted_loss}"
    # "sweep_fixmatch_{data}_lab-{active.num_labelled}_{model}_ep-{trainer.max_epochs}"
)
# naming_conv = "sweep_fixmatch_{data}_{model}_{trainer.max_epochs}_{active.num_labelled}"  # {model}"
path_to_ex_file = "src/run_training_fixmatch.py"

joint_iteration = ["active.num_labelled", "data.val_size"]


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
