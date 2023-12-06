from argparse import ArgumentParser

from launcher import ExperimentLauncher

config_dict = {
    "model": "resnet",
    "query": [
        "random",
    ],
    "data": ["isic2019"],
    "active": [
        "full_data",
    ],
    "optim": ["sgd_cosine"],
}


# Precision 16: 9,9 Gb
#
hparam_dict = {
    "data.balanced_sampling": True,
    "trainer.seed": [12345, 12346, 12347],
    "trainer.max_epochs": 200,
    "model.dropout_p": [0],
    "trainer.precision": 16,  # had some stability issues with weighted loss etc... in doubt use 32
    "model.learning_rate": [0.1, 0.01],
    "model.weight_decay": [5e-3, 5e-4, 5e-5],
    "model.use_ema": False,
    "model.small_head": [True],
    "model.weighted_loss": False,
    "data.transform_train": ["isic_randaugmentMC", "isic_train"],
    "trainer.deterministic": True,
    "trainer.num_workers": 12,
    "trainer.batch_size": 512,  # note: bigger batchsizes make training much faster!
}
naming_conv = "{data}/{active}/basic_model-{model}_drop-{model.dropout_p}_aug-{data.transform_train}_wd-{model.weight_decay}_lr-{model.learning_rate}_optim-{optim}_weighted-{model.weighted_loss}_epochs-{trainer.max_epochs}_balancsamp-{data.balanced_sampling}"

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
