from argparse import ArgumentParser

from launcher import BaseLauncher

config_dict = {
    "data": ["miotcd"],
    "model": ["imagenet_resnet18"],
}

hparam_dict = {
    "trainer.gpus": -1,
    "trainer.seed": [12345, 12346, 12347],
    "model.batch_size": 256,
    "trainer.max_epochs": 200,
    "trainer.precision": 32,  # NaN errors for fp16 with seeds 12346 & 12347
    "trainer.deterministic": True,
    "trainer.num_workers": 12,
}

naming_conv = "{data}/{model}"

path_to_ex_file = "ssl/train_simclr.py"

joint_iteration = None

if __name__ == "__main__":
    parser = ArgumentParser()
    BaseLauncher.add_argparse_args(parser)
    # parser.add_argument("--gpus", type=int, default=hparam_dict["trainer.gpus"])
    launcher_args = parser.parse_args()

    config_dict, hparam_dict = BaseLauncher.modify_params_for_args(
        launcher_args, config_dict, hparam_dict
    )

    # This has to stay here, BaseLauncher does not change this!
    if "model.load_pretrained" in hparam_dict:
        hparam_dict["model.load_pretrained"] = BaseLauncher.finalize_paths(
            hparam_dict["model.load_pretrained"],
            on_cluster=launcher_args.bsub,
        )

    launcher = BaseLauncher(
        config_dict,
        hparam_dict,
        launcher_args,
        naming_conv,
        path_to_ex_file,
        joint_iteration=joint_iteration,
    )

    if launcher_args.bsub:
        launcher.ex_call = "~/run_active_ddp.sh python"

    launcher.launch_runs()
