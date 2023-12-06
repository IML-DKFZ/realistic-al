from argparse import ArgumentParser

from launcher import ExperimentLauncher

config_dict = {
    "model": "resnet",
    "query": [
        "random",
        "entropy",
        "kcentergreedy",
        "badge",
    ],
    "data": ["cifar100"],
    "active": [
        "cifar100_low",
        "cifar100_med",
    ],
    "optim": ["sgd_fixmatch"],
}

hparam_dict = {
    "data.val_size": [500 * 5, None],
    "active.acq_size": [1500, 3000],
    "active.num_iter": [4],
    "trainer.seed": [12345, 12346, 12347],
    "trainer.deterministic": True,
    "trainer.max_epochs": 200,
    "model.dropout_p": [0],
    "model.weight_decay": [5e-4],
    "model.distr_align": True,
    "model.learning_rate": [0.03],
    "model.small_head": [True],
    "model.use_ema": False,
    "data.transform_train": [
        "cifar_basic",
    ],
    "sem_sl.eman": [False],
    "model.load_pretrained": None,
}
naming_conv = "{data}/active-{active}/fixmatch_model-{model}_drop-{model.dropout_p}_aug-{data.transform_train}_acq-{query}_ep-{trainer.max_epochs}"

joint_iteration = [
    ["data.val_size", "active", "active.acq_size"],
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

    # This is only required for BADGE
    # if launcher_args.bsub:
    #     launcher.ex_call = "~/run_active_20gb.sh python"

    launcher.launch_runs()
