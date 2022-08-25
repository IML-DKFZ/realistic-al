from argparse import ArgumentParser
from launcher import ExperimentLauncher

# Add Transformations from Randaugment and Changing of Learning Rates

config_dict = {
    "model": ["resnet"],
    "data": "cifar10_imb",
    "optim": ["sgd"],
}

load_pretrained = [
    "SSL/SimCLR/cifar10/2021-11-11_16:20:56.103061/checkpoints/last.ckpt",
    "SSL/SimCLR/cifar10/2021-11-15_10:29:02.475176/checkpoints/last.ckpt",
    "SSL/SimCLR/cifar10/2021-11-15_10:29:02.500429/checkpoints/last.ckpt",
]

hparam_dict = {
    "trainer.run_test": False,
    "model.weighted_loss": True,
    "active.num_labelled": [50, 250, 1000],
    "data.val_size": [50 * 5, 250 * 5, None],
    "model.dropout_p": [0],
    "model.learning_rate": [0.01, 0.001],  # is more stable than 0.1!
    "model.weight_decay": [5e-3, 5e-4],
    # "model.use_ema": [True, False],
    "model.small_head": False,
    "model.use_ema": False,
    "model.load_pretrained": load_pretrained,
    "trainer.max_epochs": 200,
    "trainer.seed": [12345, 12346, 12347],
    "data.transform_train": ["cifar_randaugment"],
}

naming_conv = "sweep/{data}/basic-pretrained_lab-{active.num_labelled}_{model}_ep-{trainer.max_epochs}_drop-{model.dropout_p}_lr-{model.learning_rate}_wd-{model.weight_decay}_opt-{optim}_trafo-{data.transform_train}"

path_to_ex_file = "src/run_training.py"

joint_iteration = [
    ["model.load_pretrained", "trainer.seed"],
    ["active.num_labelled", "data.val_size"],
]


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
