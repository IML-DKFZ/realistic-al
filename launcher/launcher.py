from argparse import ArgumentParser, Namespace
import os
import subprocess
from itertools import product
from copy import deepcopy

cluster_sync_call = "cluster_sync"
cluster_log_path = "/gpu/checkpoints/OE0612/c817h"
cluster_ex_call = "cluster_run"

local_ex_call = "python"
local_log_path = "/home/c817h/Documents/logs_cluster"


class BaseLauncher:
    def __init__(
        self,
        config_args: dict,
        overwrite_args: dict,
        launcher_args: Namespace,
        naming_convention: str,
        path_to_ex_file: str,
        default_struct: bool = True,
        add_name: str = "++trainer.experiment_name=",
        joint_iteration: list = None,
    ):
        """Launcher allowing fast parsing of parameters and experiments on both Cluster and the local Workstation!"""
        if default_struct:
            current_dir = os.path.dirname(os.path.realpath(__file__))
            base_dir = "/".join(current_dir.split("/")[:-1])
            path_to_ex_file = os.path.join(base_dir, path_to_ex_file)

        self.path_to_ex_file = path_to_ex_file

        self.config_args = self.check_dictionary_iterable(config_args)
        self.overwrite_args = self.check_dictionary_iterable(overwrite_args)

        self.cluster_log_path = cluster_log_path
        self.cluster_ex_call = cluster_ex_call
        self.cluster_sync_call = cluster_sync_call

        self.local_log_path = local_log_path
        self.local_ex_call = local_ex_call

        self.launcher_args = launcher_args

        # create iteration list
        self.product = self.create_product()
        self.joint_iteration = joint_iteration
        self.parsed_product = self.parse_product()

        # Naming Scheme
        self.naming_convention = naming_convention
        self.add_name = add_name

        if launcher_args.cluster:
            self.ex_call = cluster_ex_call
            self.log_path = cluster_log_path

        else:
            self.ex_call = local_ex_call
            self.log_path = local_log_path

    def sync_cluster(self):
        subprocess.call(self.cluster_sync_call, shell=True)

    def create_product(self) -> product:
        return product(
            *list(self.config_args.values()), *list(self.overwrite_args.values())
        )

    @staticmethod
    def check_dictionary_iterable(dictionary: dict):
        for key, val in dictionary.items():
            if not isinstance(val, (list, tuple)):
                dictionary[key] = [val]
        return dictionary

    def generate_name(self, config_dict: dict, param_dict: dict) -> str:
        naming_dict = self.merge_dictionaries(config_dict, param_dict)

        temp_dict = {}
        for key, val in naming_dict.items():
            key_use = self.validify_string_for_format(key)
            temp_dict[key_use] = val

        temp_naming_convention = self.validify_string_for_format(self.naming_convention)

        return temp_naming_convention.format_map(temp_dict)

    @staticmethod
    def merge_dictionaries(config_dict, param_dict):
        naming_dict = deepcopy(config_dict)
        naming_dict.update(param_dict)
        return naming_dict

    def validify_string_for_format(self, string: str):
        return string.replace(".", "")

    @staticmethod
    def add_argparse_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument("-c", "--cluster", action="store_true")
        parser.add_argument("-d", "--debug", action="store_true")
        parser.add_argument("--num_start", default=0, type=int)
        parser.add_argument("--num_end", default=-1, type=int)
        return parser

    def parse_product(self) -> list:
        # add here 1. Verifying that all values have the same length
        # 2. A way to jump over configs, where these values are different
        joint_dict = self.merge_dictionaries(self.config_args, self.overwrite_args)
        accept_dicts = self.unfold_zip_dictionary(joint_dict)

        final_product = []
        for args in deepcopy(self.product):
            config_dict = dict(
                zip(self.config_args.keys(), args[: len(self.config_args)])
            )
            param_dict = dict(
                zip(self.overwrite_args.keys(), args[len(self.config_args) :])
            )
            full_dict = self.merge_dictionaries(config_dict, param_dict)

            if not self.check_config(full_dict):
                # Check whether the the acc_dict lies in the full_dict
                # If yes, then it is is accepted for execution
                for acc_dict in accept_dicts:
                    if all(
                        full_dict.get(key, None) == val for key, val in acc_dict.items()
                    ):
                        final_product.append((config_dict, param_dict))
        return final_product

    def unfold_zip_dictionary(self, joint_dict):
        accept_dicts = [dict()]
        if self.joint_iteration is not None:
            for i, key in enumerate(self.joint_iteration):
                if i == 0:
                    check_length = len(joint_dict[key])
                    accept_dicts = [dict() for i in range(check_length)]
                assert check_length == len(joint_dict[key])
                for acc_dict, val in zip(accept_dicts, joint_dict[key]):
                    acc_dict[key] = val
        return accept_dicts

    def check_config(self, config_settings: tuple) -> bool:
        """This function returns true if no execution with this config should be executed"""
        return False

    def launch_runs(self):
        """Execute all specified Runs"""
        self.prepare_launch()
        for i, (config_dict, param_dict) in enumerate(self.parsed_product):
            counter = i + 1
            if counter <= self.launcher_args.num_start or (
                counter > self.launcher_args.num_end and self.launcher_args.num_end > 0
            ):
                continue
            print(f"Launch: {counter}/{len(self.parsed_product)}")

            config_args = self.dict_to_arg(config_dict)
            param_args = self.dict_to_arg(param_dict, prefix="++")

            experiment_name = self.generate_name(config_dict, param_dict)
            experiment_arg = self.add_name + experiment_name

            full_args = config_args + param_args + " " + experiment_arg
            launch_command = f"{self.ex_call} {self.path_to_ex_file} {full_args}"

            print(launch_command)
            if not self.launcher_args.debug:
                subprocess.call(launch_command, shell=True)

    def prepare_launch(self):
        if self.launcher_args.cluster and self.launcher_args.debug is False:
            self.sync_cluster()

    @staticmethod
    def dict_to_arg(dict, prefix="", key_to_arg="="):
        argument_string = ""
        for key, value in dict.items():
            argument_string += f"{prefix}{key}{key_to_arg}{value} "
        return argument_string

    @staticmethod
    def finalize_paths(paths, on_cluster: bool):
        """Change all Paths to PreTrained Models according to Environment.
        paths should start, where log_path ends."""
        if on_cluster:
            log_path = cluster_log_path
        else:
            log_path = local_log_path
        if not isinstance(paths, (list, tuple)):
            paths = [paths]

        out = []
        for path in paths:
            out.append(os.path.join(log_path, path))
        return out


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    BaseLauncher.add_argparse_args(parser)
    args = parser.parse_args()
    configs = {"data": ["cifar10"], "model": ["resnet18", "vgg"], "exp": "test"}

    load_pretrained = (
        "SSL/SimCLR/cifar10/2021-11-11_16:20:56.103061/checkpoints/last.ckpt",
        "SSL/SimCLR/cifar10/2021-11-15_10:29:02.475176/checkpoints/last.ckpt",
    )

    load_pretrained = BaseLauncher.finalize_paths(
        load_pretrained, on_cluster=args.cluster
    )

    hparams = {
        "model.dropout_p": [0, 0.5],
        "trainer.name": ["1234", "5678"],
        "model.load_pretrained": load_pretrained,
    }

    if args.cluster:
        hparams["trainer.progress_bar_refresh_rate"] = 0

    naming_conv = "{trainer.name}_test_v2"
    launcher = BaseLauncher(
        configs,
        hparams,
        args,
        naming_conv,
        "src/run_training_fixmatch.py",
        joint_iteration=["model.dropout_p", "model"],
        # joint_iteration=None,
    )
    launcher.launch_runs()
