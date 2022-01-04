from argparse import ArgumentParser
from launcher import ExperimentLauncher

config_dict = {
    "model": "resnet",
    "query": ["random", "entropy", "kcentergreedy", "bald"],
    "data": ["cifar10"],  # , "cifar100"],
    "active": ["standard"],  # did not run! "standard_250", "cifar10_low_data"
    "optim": ["sgd"],
}

# Pretrained models from Baseline Pytorch Lightning Bolts - for final results, use own version
load_pretrained = [
    "SSL/SimCLR/cifar10/2021-11-11_16:20:56.103061/checkpoints/last.ckpt",
    "SSL/SimCLR/cifar10/2021-11-15_10:29:02.475176/checkpoints/last.ckpt",
    "SSL/SimCLR/cifar10/2021-11-15_10:29:02.500429/checkpoints/last.ckpt",
]
hparam_dict = {
    "trainer.seed": [12345, 12346, 12347],
    "model.dropout_p": [0, 0.5],
    "model.learning_rate": [0.001],
    "model.freeze_encoder": [True, False],
    "model.finetune": [False],
    "model.use_ema": False,
    "model.load_pretrained": load_pretrained,
}
naming_conv = "{data}_{active}_{query}_drop-{model.dropout_p}_pretrained_goodfit"


joint_iteration = ["model.load_pretrained", "trainer.seed"]

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
