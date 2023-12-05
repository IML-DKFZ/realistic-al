from argparse import ArgumentParser

from launcher import ExperimentLauncher

config_dict = {
    "model": "resnet",
    "query": [
        "random",
    ],
    "data": ["cifar10", "cifar100"],
    # "data": ["cifar100"],  # , "cifar100"],
    "active": [
        "full_data",
    ],
    "optim": ["sgd"],
}

hparam_dict = {
    "trainer.seed": [12345, 12346, 12347],
    "trainer.max_epochs": 80,  # Think about this before commiting (or sweep!)
    "model.dropout_p": [0, 0.5],
    "model.learning_rate": [0.001],
    "model.freeze_encoder": [False],  # possibly add True
    # "model.finetune": [True],
    "model.use_ema": False,
    "model.load_pretrained": True,
    "data.transform_train": [
        "cifar_basic",
        "cifar_randaugment",
    ],
    "model.small_head": [False],
    "trainer.precision": 32,
}
naming_conv = (
    "{data}/{active}/basic_model-{model}_drop-{model.dropout_p}_aug-{data.transform_train}_acq-{query}_ep-{trainer.max_epochs}"
    # "active_basic_{data}_set-{active}_{model}_acq-{query}_ep-{trainer.max_epochs}"
)

joint_iteration = ["model.load_pretrained", "trainer.seed"]

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
