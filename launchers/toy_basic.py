from argparse import ArgumentParser
from launcher import ExperimentLauncher

config_dict = {
    "model": "bayesian_mlp",
    "query": [
        "random",
        "entropy",
        "bald",
        "batchbald",
        "variationratios",
        # "kcentergreedy", # TODO: make k-CenterGreedy run
    ],
    "data": ["toy_moons", "toy_circles"],
    "active": "toy_two_moons",
}

hparam_dict = {
    "trainer.seed": [12345 + i for i in range(10)],
    "trainer.max_epochs": 40,  # later change to 40?
    "trainer.vis_callback": False,
    "model.dropout_p": [0, 0.25],  # dropout 0.5 does not work
    "model.use_bn": False,  # for low data regime batchnorm does not work well!
}
naming_conv = "{data}/active_basic_set-{active}_{model}_dop-{model.dropout_p}_acq-{query}_ep-{trainer.max_epochs}"

joint_iteration = None

path_to_ex_file = "src/main_toy.py"

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
