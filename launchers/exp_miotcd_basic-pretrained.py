from argparse import ArgumentParser
from launcher import ExperimentLauncher

config_dict = {
    "model": ["resnet"],
    "data": "miotcd",
    "active": ["miotcd_low", "miotcd_med", "miotcd_high"],  # standard
    "query": ["random", "entropy", "kcentergreedy", "bald"],
    "optim": ["sgd"],
}

load_pretrained = [
    "SSL/miotcd/imagenet_resnet18/2022-09-06_11-33-02-630425/checkpoints/last.ckpt",  # seed = 12345
    "SSL/miotcd/imagenet_resnet18/2022-09-06_14-12-43-497039/checkpoints/last.ckpt",  # seed = 12346
    "SSL/miotcd/imagenet_resnet18/2022-09-06_14-12-43-189723/checkpoints/last.ckpt",  # seed = 12347
]

num_classes = 11
hparam_dict = {
    "trainer.run_test": False,
    "data.val_size": [num_classes * 5 * 5, num_classes * 25 * 5, num_classes * 100 * 5],
    # "active.num_labelled": [200, 800],
    # "data.val_size": [1000, None],
    "model.dropout_p": [0, 0, 0, 0.5],
    "model.learning_rate": [
        0.001
    ],  # 0,01 is omitted due to bad performance on every dataset!
    "model.weight_decay": [5e-3],
    "model.use_ema": False,
    "model.small_head": [False],
    "model.weighted_loss": True,
    "model.load_pretrained": load_pretrained,
    "model.freeze_encoder": False,
    "trainer.max_epochs": 80,
    "trainer.batch_size": 128,
    "trainer.num_workers": 10,
    "trainer.seed": [12345, 12346, 12347],
    "data.transform_train": ["imagent_randaug", "imagenet_train", "imagenet_train"],
}

joint_iteration = [
    ["query", "model.dropout_p"],
    ["trainer.seed", "model.load_pretrained"],
    ["active", "data.transform_train", "data.val_size"],
]

naming_conv = "sweep/{data}/basic-pretrained_model-{model}_drop-{model.dropout_p}_aug-{data.transform_train}_acq-{query}_ep-{trainer.max_epochs}_freeze-{model.freeze_encoder}_smallhead-{model.small_head}"

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
