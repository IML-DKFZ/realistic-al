from argparse import ArgumentParser

from launcher import ExperimentLauncher

config_dict = {
    "model": ["resnet"],
    "data": "miotcd",
    "active": ["miotcd_low"],
    "query": [
        # "random",
        # "entropy",
        # "kcentergreedy",
        # "bald",
        "batchbald",
    ],
    "optim": ["sgd"],
}

load_pretrained = [
    "SSL/miotcd/imagenet_resnet18/2022-09-06_11-33-02-630425/checkpoints/last.ckpt",  # seed = 12345
    "SSL/miotcd/imagenet_resnet18/2022-09-06_14-12-43-497039/checkpoints/last.ckpt",  # seed = 12346
    "SSL/miotcd/imagenet_resnet18/2022-09-06_14-12-43-189723/checkpoints/last.ckpt",  # seed = 12347
]

num_classes = 11
hparam_dict = {
    "active.m": 50000,
    "trainer.run_test": True,
    "data.val_size": [num_classes * 5 * 5],
    "model.dropout_p": [0.5],
    "model.learning_rate": [0.001],
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
    "data.transform_train": ["imagenet_randaug"],
}

joint_iteration = [
    ["trainer.seed", "model.load_pretrained"],
]

naming_conv = "{data}/active-{active}-batchbald/basic-pretrained_model-{model}_drop-{model.dropout_p}_aug-{data.transform_train}_acq-{query}_ep-{trainer.max_epochs}_freeze-{model.freeze_encoder}_smallhead-{model.small_head}"

path_to_ex_file = "src/main.py"


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
