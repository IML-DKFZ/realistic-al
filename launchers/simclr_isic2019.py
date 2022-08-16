from argparse import ArgumentParser
from launcher import BaseLauncher

config_dict = {
    "data": ["isic2019"],
    "model": ["cifar_resnet18"],
}

hparam_dict = {
    "trainer.num_gpus": 2,
    "trainer.seed": [12345, 12346, 12347],
    "trainer.max_epochs": 1000,
}

naming_conv = "{data}/{model}"

path_to_ex_file = "skripts/train_simclr.py"

joint_iteration = None

if __name__ == "__main__":
    parser = ArgumentParser()
    BaseLauncher.add_argparse_args(parser)
    launcher_args = parser.parse_args()

    config_dict, hparam_dict = BaseLauncher.modify_params_for_args(
        launcher_args, config_dict, hparam_dict
    )

    # This has to stay here, BaseLauncher does not change this!
    if "model.load_pretrained" in hparam_dict:
        hparam_dict["model.load_pretrained"] = BaseLauncher.finalize_paths(
            hparam_dict["model.load_pretrained"], on_cluster=launcher_args.cluster
        )

    launcher = BaseLauncher(
        config_dict,
        hparam_dict,
        launcher_args,
        naming_conv,
        path_to_ex_file,
        joint_iteration=joint_iteration,
    )
    if launcher_args.cluster:
        launcher.ex_call = "cluster_run --launcher run_active_2gpu.sh"

    launcher.launch_runs()
