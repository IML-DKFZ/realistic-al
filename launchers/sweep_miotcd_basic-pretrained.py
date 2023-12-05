from argparse import ArgumentParser

from launcher import ExperimentLauncher

config_dict = {
    "model": ["resnet"],
    "data": "miotcd",
    "optim": ["sgd"],
    # "optim": "sgd_cosine",  # update
}

load_pretrained = [
    "SSL/miotcd/imagenet_resnet18/2022-09-06_11-33-02-630425/checkpoints/last.ckpt",  # seed = 12345
    "SSL/miotcd/imagenet_resnet18/2022-09-06_14-12-43-497039/checkpoints/last.ckpt",  # seed = 12346
    "SSL/miotcd/imagenet_resnet18/2022-09-06_14-12-43-189723/checkpoints/last.ckpt",  # seed = 12347
]

num_classes = 11
hparam_dict = {
    "trainer.run_test": False,
    "active.num_labelled": [num_classes * 5, num_classes * 25, num_classes * 100],
    "data.val_size": [num_classes * 5 * 5, num_classes * 25 * 5, num_classes * 100 * 5],
    # "active.num_labelled": [200, 800],
    # "data.val_size": [1000, None],
    "model.dropout_p": [0],
    "model.learning_rate": [
        0.001,
        0.01,
    ],  # 0,01 is omitted due to bad performance on every dataset!
    "model.weight_decay": [5e-3, 5e-4],  # udpate to 5e-3
    "model.use_ema": False,
    "model.small_head": [False],
    "model.weighted_loss": True,
    "model.load_pretrained": load_pretrained,
    "trainer.max_epochs": 80,
    "trainer.batch_size": 128,
    # "trainer.batch_size": 512,  # update
    "trainer.precision": 16,
    "trainer.num_workers": 10,
    "trainer.seed": [12345, 12346, 12347],
    "data.transform_train": [
        "imagenet_train",
        "imagenet_randaugMC",
    ],
}

joint_iteration = [
    ["active.num_labelled", "data.val_size"],
    ["trainer.seed", "model.load_pretrained"],
]

naming_conv = "sweep/{data}/basic-pretrained_lab-{active.num_labelled}_{model}_ep-{trainer.max_epochs}_drop-{model.dropout_p}_lr-{model.learning_rate}_wd-{model.weight_decay}_opt-{optim}_trafo-{data.transform_train}"

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
