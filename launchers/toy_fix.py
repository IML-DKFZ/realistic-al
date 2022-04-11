from argparse import ArgumentParser
from launcher import ExperimentLauncher

config_dict = {
    "model": [
        "bayesian_mlp"
    ],  # , "bayesian_mlp_deep"], TODO: fix freezing for deep model!
    # "model": "bayesian_mlp_deep",
    # "query": ["random"],
    "query": ["random", "entropy", "bald", "batchbald"],
    "data": "toy_moons",
    "active": "toy_two_moons",
}

hparam_dict = {
    "trainer.seed": [12345, 12346, 12347],
    # "trainer.seed": 12345,
    "trainer.max_epochs": 40,
    "trainer.vis_callback": True,
    "model.weight_decay": [0.01],
    "model.dropout_p": [0.25],  # dropout 0.5 does not work
    "sem_sl.lambda_u": [3],
}
naming_conv = "{data}/active_fixmatch_set-{active}_{model}_dop-{model.dropout_p}_acq-{query}_ep-{trainer.max_epochs}"
# naming_conv = "{data}_sweeps/fixmatch_{model}_drop-{model.dropout_p}_wd-{model.weight_decay}_lambda-{sem_sl.lambda_u}"

joint_iteration = None

path_to_ex_file = "src/main_toy_fixmatch.py"

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
