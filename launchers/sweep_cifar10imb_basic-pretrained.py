from argparse import ArgumentParser

from launcher import ExperimentLauncher

# Add Transformations from Randaugment and Changing of Learning Rates

config_dict = {
    "model": ["resnet"],
    "data": "cifar10_imb",
    # "optim": ["sgd"],
    "optim": ["sgd_cosine"],
}

load_pretrained = [
    "SSL/cifar10_imb/cifar_resnet18/2022-07-15_11-31-31-278397/checkpoints/last.ckpt",  # seed = 12345
    "SSL/cifar10_imb/cifar_resnet18/2022-07-15_11-31-34-631836/checkpoints/last.ckpt",  # seed = 12346
    "SSL/cifar10_imb/cifar_resnet18/2022-07-15_11-31-34-632366/checkpoints/last.ckpt",  # seed = 12347
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
    "trainer.max_epochs": 80,
    "trainer.seed": [12345, 12346, 12347],
    "data.transform_train": ["cifar_basic", "cifar_randaugment"],
    # "data.transform_train": ["cifar_basic", "cifar_randaugmentMC",],
    "trainer.precision": 16,
    # "trainer.batch_size": 1024,
    # "trainer.batch_size": 128,
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
