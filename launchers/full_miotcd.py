from argparse import ArgumentParser
from launcher import ExperimentLauncher

config_dict = {
    "model": "resnet",
    "query": ["random",],
    "data": ["miotcd"],
    "active": ["full_data",],
    "optim": ["sgd"],
}

hparam_dict = {
    "trainer.seed": [12345, 12346, 12347],
    "trainer.max_epochs": 200,
    "model.dropout_p": [0],
    "model.learning_rate": [0.1, 0.01],
    "model.weight_decay": [5e-3, 5e-4],
    "model.use_ema": False,
    "model.small_head": [True],
    "model.weighted_loss": True,
    "trainer.max_epochs": 80,
    "trainer.seed": [12345, 12346, 12347],
    "data.transform_train": ["imagenet_train", "imagenet_randaugment"],
    "trainer.deterministic": True,
    "trainer.num_workers": 12,
    "trainer.precision": 16,
    "trainer.batch_size": 512,  # note: batchsize of 128 makes trainings much faster!
}
naming_conv = "{data}/{active}/basic_model-{model}_drop-{model.dropout_p}_aug-{data.transform_train}_wd-{model.weight_decay}_lr-{model.learning_rate}"

joint_iteration = None

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
