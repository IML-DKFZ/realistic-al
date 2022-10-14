from argparse import ArgumentParser
from launcher import ExperimentLauncher

config_dict = {
    "model": "resnet_fixmatch",
    "data": "isic2019",
    "optim": ["sgd_fixmatch"],
}

hparam_dict = {
    "trainer.run_test": False,
    "active.num_labelled": [40, 200],
    "data.val_size": [200, 1000],
    "model.dropout_p": [0],
    "model.learning_rate": [0.3, 0.03],
    "model.weight_decay": [5e-3, 5e-4],
    "model.use_ema": False,
    "model.small_head": [True],
    "model.weighted_loss": True,
    # "trainer.max_epochs": 200,
    "trainer.max_epochs": 50,
    "trainer.seed": [12345, 12346, 12347],
    "data.transform_train": ["isic_train"],
    "trainer.precision": 16,  # prec 16 is 1.5 times faster than prec 32 (NaN errors for prec16)
}

joint_iteration = [["active.num_labelled", "data.val_size"]]

naming_conv = "sweep/{data}/fixmatch_lab-{active.num_labelled}_{model}_ep-{trainer.max_epochs}_drop-{model.dropout_p}_lr-{model.learning_rate}_wd-{model.weight_decay}_opt-{optim}_trafo-{data.transform_train}_wloss-{model.weighted_loss}"

path_to_ex_file = "src/run_training_fixmatch.py"


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

    if launcher_args.cluster:
        launcher.ex_call = "cluster_run --launcher run_active_25gb.sh"

    launcher.launch_runs()
