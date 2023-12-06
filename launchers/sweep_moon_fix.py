from argparse import ArgumentParser

from launcher import ExperimentLauncher

config_dict = {
    # "model": ["bayesian_mlp", "bayesian_mlp_deep"],
    "model": "bayesian_mlp_deep",
    "query": ["random"],
    "data": "toy_moons",
    "active": "toy_two_moons",
}

hparam_dict = {
    "trainer.seed": [12345 + i for i in range(10)],
    "trainer.precision": 32,
    "trainer.max_epochs": 40,
    "active.num_labelled": [6],
    "active.num_iter": 5,
    "active.acq_size": 12,
    "trainer.vis_callback": False,
    "model.weight_decay": [0, 0.01, 0.001],  # 0.1 is much too high!
    "model.dropout_p": [0, 0.25],  # [0.25],  # dropout 0.5 does not work
    "model.use_ema": False,  # [True, False],
    "sem_sl.lambda_u": [1, 3, 5],
}
naming_conv = "{data}_sweeps/fixmatch_{model}_drop-{model.dropout_p}_wd-{model.weight_decay}_lambda-{sem_sl.lambda_u}_ema-{model.use_ema}"

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
