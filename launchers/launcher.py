from argparse import ArgumentParser, Namespace
import os
import subprocess
from itertools import product
from copy import deepcopy
from config_launcher import get_pretrained_arch
import yaml

cluster_sync_call = "cluster_sync"
cluster_log_path = "/dkfz/cluster/gpu/checkpoints/OE0612/c817h"
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
        """Launcher allowing fast parsing of parameters and experiments on both Cluster and the local Workstation"""
        if default_struct:
            # TODO: Change __file__ since this will always lead to the directory here where launcher.py is located!
            current_dir = os.path.dirname(os.path.realpath(__file__))
            base_dir = "/".join(current_dir.split("/")[:-1])
            path_to_ex_file = os.path.join(base_dir, path_to_ex_file)

        self.path_to_ex_file = path_to_ex_file

        self.config_args = self.make_dictionary_iterable(config_args)
        self.overwrite_args = self.make_dictionary_iterable(overwrite_args)

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
    def make_dictionary_iterable(dictionary: dict):
        for key, val in dictionary.items():
            if not isinstance(val, (list, tuple)):
                dictionary[key] = [val]
        return dictionary

    def generate_name(self, config_dict: dict, param_dict: dict) -> str:
        """Generates name by formatting the naming_convention"""
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

    @staticmethod
    def modify_params_for_args(
        launcher_args: Namespace, config_dict: dict, hparam_dict: dict
    ):
        if launcher_args.cluster:
            hparam_dict["trainer.progress_bar_refresh_rate"] = 0
        return config_dict, hparam_dict

    def parse_product(self) -> list:
        """Create a list with all combinations that should be launched."""
        # add here 1. Verifying that all values have the same length
        # 2. A way to jump over configs, where these values are different
        joint_dict = self.merge_dictionaries(self.config_args, self.overwrite_args)
        accept_dicts = [dict()]
        accept_dicts_arr = []
        if isinstance(self.joint_iteration[0], (list, tuple)):
            for joint_iteration in self.joint_iteration:
                accept_dicts_arr.append(
                    self.unfold_zip_dictionary(joint_dict, joint_iteration)
                )

        else:
            accept_dicts_arr.append(
                self.unfold_zip_dictionary(joint_dict, self.joint_iteration)
            )

        # full_dicts = []
        config_dicts = []
        param_dicts = []

        for args in deepcopy(self.product):
            config_dict = dict(
                zip(self.config_args.keys(), args[: len(self.config_args)])
            )
            param_dict = dict(
                zip(self.overwrite_args.keys(), args[len(self.config_args) :])
            )
            config_dicts.append(config_dict)
            param_dicts.append(param_dict)

        for accept_dicts in accept_dicts_arr:
            config_dicts_new = []
            param_dicts_new = []

            for config_dict, param_dict in zip(config_dicts, param_dicts):
                full_dict = self.merge_dictionaries(config_dict, param_dict)
                if not self.skip_config(full_dict):
                    full_dict = self.merge_dictionaries(config_dict, param_dict)
                    # Check whether the the acc_dict lies in the full_dict
                    # If yes, then it is is accepted for execution
                    for acc_dict in accept_dicts:
                        if (
                            all(
                                full_dict.get(key, None) == val
                                for key, val in acc_dict.items()
                            )
                            # and (config_dict, param_dict) not in final_product
                        ):
                            config_dicts_new.append(config_dict)
                            param_dicts_new.append(param_dict)
            config_dicts = config_dicts_new
            param_dicts = param_dicts_new

        final_product = []
        for config_dict, param_dict in zip(config_dicts, param_dicts):
            if (config_dict, param_dict) not in final_product:
                final_product.append((config_dict, param_dict))
        return final_product

    def unfold_zip_dictionary(self, joint_dict, joint_iteration):
        """Creates a list of dictionaries with combinations that
        should be launched according to joint_iteration.
        The format of the output is identical to the one obtained by itertools!"""
        accept_dicts = [dict()]
        if joint_iteration is not None:
            for i, key in enumerate(joint_iteration):
                if i == 0:
                    check_length = len(joint_dict[key])
                    accept_dicts = [dict() for i in range(check_length)]
                assert check_length == len(joint_dict[key])
                for acc_dict, val in zip(accept_dicts, joint_dict[key]):
                    acc_dict[key] = val
        return accept_dicts

    def skip_config(self, config_settings: dict) -> bool:
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

    def skip_config(self, config_settings: dict):
        return False

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

    @staticmethod
    def access_config(config_name):
        raise NotImplementedError


class ExperimentLauncher(BaseLauncher):
    def skip_config(self, config_settings: dict):
        if set(["sem_sl.eman", "model.use_ema"]).issubset(config_settings.keys()):
            if config_settings["sem_sl.eman"] and (
                config_settings["model.use_ema"] is False
            ):
                return True

        if set(["model.finetune", "model.freeze_encoder"]).issubset(
            config_settings.keys()
        ):
            if (
                config_settings["model.finetune"]
                and config_settings["model.freeze_encoder"]
            ):
                return True

        if set(["query", "model.dropout_p"]).issubset(config_settings.keys()):
            if (
                config_settings["query"] in ["bald", "batchbald"]
                and config_settings["model.dropout_p"] == 0
            ):
                return True

        return False

    @staticmethod
    def access_config_value(
        key: str,
        config_name: str,
        config_dict: dict,
        hparam_dict: dict,
        default_struct: bool = True,
        file_ending: str = ".yaml",
    ):
        """Returns the value of a predefined config in hydra, given the key etc."""
        full_key = "{config_name}.{key}"
        if full_key in hparam_dict:
            return hparam_dict[full_key]

        if default_struct:
            # print(f"{key=}")
            # print(f"{config_dict=}")
            # print(f"{config_name=}")
            # print(config_dict[config_name])
            current_dir = os.path.dirname(os.path.realpath(__file__))
            base_dir = "/".join(current_dir.split("/")[:-1])
            filename = config_dict[config_name]
            print(f"{filename=}")
            if isinstance(filename, (list, tuple)):
                if len(filename) > 1:
                    raise NotImplementedError(
                        "Currently only parameter of one file can be loaded in this fashion."
                    )
                filename = filename[0]
            path_to_config = os.path.join(
                base_dir, "src", "config", config_name, filename + file_ending,
            )
            print(path_to_config)
            assert os.path.isfile(
                path_to_config
            )  # There exists no such File, or file ending is not correct
            with open(path_to_config, "r") as stream:
                # try:
                value = yaml.safe_load(stream)[key]
                # except yaml.YAMLError as exc:
                #     print(exc)
            return value

        raise NotImplementedError

    @staticmethod
    def modify_params_for_args(
        launcher_args: Namespace, config_dict: dict, hparam_dict: dict
    ):
        config_dict, hparam_dict = BaseLauncher.modify_params_for_args(
            launcher_args, config_dict, hparam_dict
        )
        load_arch = "model.load_pretrained"
        if load_arch in hparam_dict:
            if hparam_dict["model.load_pretrained"] is True:
                model_type = ExperimentLauncher.access_config_value(
                    "name", "model", config_dict, hparam_dict
                )
                dataset = ExperimentLauncher.access_config_value(
                    "name", "data", config_dict, hparam_dict
                )
                hparam_dict["model.load_pretrained"] = [
                    get_pretrained_arch(dataset, model_type, seed).ckpt_path
                    for seed in hparam_dict["trainer.seed"]
                ]

            if hparam_dict[load_arch] is not None:
                hparam_dict[
                    "model.load_pretrained"
                ] = ExperimentLauncher.finalize_paths(
                    hparam_dict["model.load_pretrained"],
                    on_cluster=launcher_args.cluster,
                )

            if hparam_dict[load_arch] is None:
                hparam_dict["model.load_pretrained"] = "Null"

        return config_dict, hparam_dict

    @staticmethod
    def dict_to_arg(dict, prefix="", key_to_arg="="):
        """Base with Change for Translation from None to Null for Hydra"""
        argument_string = ""
        for key, value in dict.items():
            if value == None:
                argument_string += f"{prefix}{key}{key_to_arg}Null "
            else:
                argument_string += f"{prefix}{key}{key_to_arg}{value} "
        return argument_string


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    ExperimentLauncher.add_argparse_args(parser)
    launcher_args = parser.parse_args()
    config_dict = {"data": ["cifar10"], "model": ["resnet18", "vgg"], "exp": "test"}

    load_pretrained = (
        "SSL/SimCLR/cifar10/2021-11-11_16:20:56.103061/checkpoints/last.ckpt",
        "SSL/SimCLR/cifar10/2021-11-15_10:29:02.475176/checkpoints/last.ckpt",
    )

    load_pretrained = ExperimentLauncher.finalize_paths(
        load_pretrained, on_cluster=launcher_args.cluster
    )

    hparam_dict = {
        "model.dropout_p": [0, 0.5],
        "trainer.name": ["1234", "5678"],
        "model.load_pretrained": load_pretrained,
    }

    config_dict, hparam_dict = ExperimentLauncher.modify_params_for_args(
        launcher_args, config_dict, hparam_dict
    )

    naming_conv = "{trainer.name}_test_v2"
    launcher = ExperimentLauncher(
        config_dict,
        hparam_dict,
        launcher_args,
        naming_conv,
        "src/run_training_fixmatch.py",
        joint_iteration=["model.dropout_p", "model"],
        # joint_iteration=None,
    )
    launcher.launch_runs()
