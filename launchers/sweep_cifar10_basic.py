from argparse import ArgumentParser
from launcher import ExperimentLauncher

# Add Transformations from Randaugment and Changing of Learning Rates

# config_dict = {
#     "model": ["resnet", "wideresnet-cifar10"],
#     "data": "cifar10",
#     "optim": ["sgd"],
# }
config_dict = {
    "model": ["resnet"],
    "data": "cifar10",
    "optim": ["sgd"],
}

hparam_dict = {
    "active.num_labelled": [50, 500, 1000, 5000],
    "data.val_size": [250, 2500, None, None],
    "model.dropout_p": [0, 0.5],
    "model.learning_rate": [0.1, 0.01],  # is more stable than 0.1!
    "model.use_ema": False,
    # "model.finetune": [True, False],
    # "model.freeze_encoder": [True, False],
    "model.small_head": [True, False],
    "trainer.max_epochs": 80,
    "trainer.seed": [12345, 12346, 12347],
    "data.transform_train": ["cifar_randaugment"],
}

joint_iteration = ["model.load_pretrained", "trainer.seed"]

# naming_conv = "sweep_basic-pretrained_{data}_lab-{active.num_labelled}_{model}_ep-{trainer.max_epochs}"
naming_conv = "sweep/{data}/basic-pretrained_lab-{active.num_labelled}_{model}_ep-{trainer.max_epochs}_drop-{model.dropout_p}_lr-{model.learning_rate}_smallhead{model.small_head}"


# hparam_dict = {
#     "active.num_labelled": [40, 500, 1000, 5000],
#     "model.dropout_p": [0, 0.5],
#     "model.learning_rate": 0.01,  # is more stable than 0.1!
#     "model.finetune": [True, False],
#     "model.freeze_encoder": [True, False],
#     # "model.use_ema": [True, False],
#     "model.use_ema": False,
#     "trainer.max_epochs": 200,
#     "trainer.seed": [12345],  # , 12346, 12347],
#     "data.transform_train": [
#         "cifar_basic",
#         "cifar_randaugment",  # this should perform better across the board!
#     ],
# }

# naming_conv = (
#     "sweep_{data}/basic/model-{model}_lab-{active.num_labelled}_ep-{trainer.max_epochs}"
#     # "sweep_basic_{data}_lab-{active.num_labelled}_{model}_ep-{trainer.max_epochs}"
# )
path_to_ex_file = "src/run_training.py"

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
