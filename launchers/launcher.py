import os
import subprocess
from argparse import ArgumentParser, Namespace
from copy import deepcopy
from itertools import product
from typing import Any, Dict, Iterable, List, Tuple, Union

import yaml
from config_launcher import get_pretrained_arch

### Running ###
### IMPLEMENTATION ###
# Add own values here

# TODO: os.getenv($EXPERIMENT_ROOT)
CLUSTER_LOG_PATH = "/dkfz/cluster/gpu/checkpoints/OE0612/c817h"
BSUB_EX_CALL = "~/run_active.sh python"

LOCAL_EX_CALL = "python"
# TODO: os.getenv($EXPERIMENT_ROOT)
LOCAL_LOG_PATH = "/home/c817h/Documents/logs_cluster"


class BaseLauncher:
    def __init__(
        self,
        config_args: Dict[str, List[Any]],
        overwrite_args: Dict[str, List[Any]],
        launcher_args: Namespace,
        naming_convention: str,
        path_to_ex_file: str,
        default_struct: bool = True,
        add_name: str = "++trainer.experiment_name=",
        joint_iteration: List[List[str]] = None,
    ):
        """Launcher for multiple experiments via python subprocesses allowing different execution calls for local deployment or cluster.
        It produces a cartesian product of concrete configs over config_args and overwrite_args dicts.
        It can handle config_args and overwrite_args differently with regard to how they are prducted to CLI arguments.

        Args:
            config_args (Dict[List[Any]): values for config of launches.
            overwrite_args (Dict[List[Any]]): values for config of launches.
            launcher_args (Namespace): config for launcher execution.
            naming_convention (str): rules to obtain final names from config_args and overwrite_args.
            path_to_ex_file (str): path to file for execution.
            default_struct (bool, optional): If true path_to_ex_file is relative from base of directory else it is absolute. Defaults to True.
            add_name (str, optional): flag which launcher uses to give experiment names. Defaults to "++trainer.experiment_name=".
            joint_iteration (List[List[str]], optional): keys inside config_dict and overwrite_args that are iterated jointly over. Defaults to None.
        """
        if default_struct:
            # TODO: Change __file__ since this will always lead to the directory here where launcher.py is located!
            current_dir = os.path.dirname(os.path.realpath(__file__))
            base_dir = "/".join(current_dir.split("/")[:-1])
            path_to_ex_file = os.path.join(base_dir, path_to_ex_file)

        self.path_to_ex_file = path_to_ex_file

        self.config_args = self.make_dictionary_iterable(config_args)
        self.overwrite_args = self.make_dictionary_iterable(overwrite_args)

        self.launcher_args = launcher_args

        # create iteration list
        self.product = self.create_product()
        self.joint_iteration = joint_iteration
        self.parsed_product = self.parse_product()

        # Naming Scheme
        self.naming_convention = naming_convention
        self.add_name = add_name

        if launcher_args.bsub:
            self.ex_call = BSUB_EX_CALL
            self.log_path = CLUSTER_LOG_PATH

        else:
            self.ex_call = LOCAL_EX_CALL
            self.log_path = LOCAL_LOG_PATH

    def create_product(self) -> product:
        """Return cartesian product of the values dicts config_args and overwrite_args.

        Returns:
            product: cartesian product.
        """
        return product(
            *list(self.config_args.values()), *list(self.overwrite_args.values())
        )

    @staticmethod
    def make_dictionary_iterable(dictionary: Dict) -> Dict[str, Iterable]:
        for key, val in dictionary.items():
            if not isinstance(val, (list, tuple)):
                dictionary[key] = [val]
        return dictionary

    def generate_name(self, config_dict: Dict, param_dict: Dict) -> str:
        """Generates name by format mapping of combined dict into the naming convention.

        Args:
            config_dict (Dict):
            param_dict (Dict): _description_

        Returns:
            str: _description_
        """
        naming_dict = self.merge_dictionaries(config_dict, param_dict)

        temp_dict = {}
        for key, val in naming_dict.items():
            key_use = self.validify_string_for_format(key)
            temp_dict[key_use] = val

        temp_naming_convention = self.validify_string_for_format(self.naming_convention)

        return temp_naming_convention.format_map(temp_dict)

    @staticmethod
    def merge_dictionaries(config_dict: Dict, param_dict: Dict) -> Dict:
        """Returns a new instance of dictionary consisting of both key value pairs

        Args:
            config_dict (Dict): dict 1
            param_dict (Dict): dict 2

        Returns:
            Dict: dict 1 + dict 2
        """
        naming_dict = deepcopy(config_dict)
        naming_dict.update(param_dict)
        return naming_dict

    @staticmethod
    def validify_string_for_format(string: str) -> str:
        """Removes all points in string to allow it being inside the name of a directory.

        Args:
            string (str): directoryname

        Returns:
            str: directoryname with `.` replaced by ``
        """
        return string.replace(".", "")

    @staticmethod
    def add_argparse_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument("-d", "--debug", action="store_true")
        parser.add_argument("--num_start", default=0, type=int)
        parser.add_argument("--num_end", default=-1, type=int)
        parser.add_argument(
            "-b", "--bsub", action="store_true", help="Executes the script via bsub"
        )
        return parser

    @staticmethod
    def modify_params_for_args(
        launcher_args: Namespace, config_dict: Dict, hparam_dict: Dict
    ) -> Tuple[Dict, Dict]:
        """Modify config_dict and hparam_dict according to launcher_args.
        When executed on cluster, `trainer.progress_bar_refresh_rate` is set to 0.

        Args:
            launcher_args (Namespace): arguments of launcher class.
            config_dict (Dict): config values for launches.
            hparam_dict (Dict): hparam values for launches.

        Returns:
            Tuple[Dict, Dict]: New (config_dict, hparam_dict)
        """
        if launcher_args.bsub:
            hparam_dict["trainer.progress_bar_refresh_rate"] = 0
        return config_dict, hparam_dict

    def parse_product(self) -> list:
        """Create a list with all combinations that should be launched."""
        # add here 1. Verifying that all values have the same length
        # 2. A way to jump over configs, where these values are different
        joint_dict = self.merge_dictionaries(self.config_args, self.overwrite_args)
        accept_dicts = [{}]
        accept_dicts_arr = []
        if self.joint_iteration is None:
            accept_dicts_arr.append(
                self.unfold_zip_dictionary(joint_dict, self.joint_iteration)
            )
        elif isinstance(self.joint_iteration[0], (list, tuple)):
            for joint_iteration in self.joint_iteration:
                accept_dicts_arr.append(
                    self.unfold_zip_dictionary(joint_dict, joint_iteration)
                )
        else:
            accept_dicts_arr.append(
                self.unfold_zip_dictionary(joint_dict, self.joint_iteration)
            )
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
                        if all(
                            full_dict.get(key, None) == val
                            for key, val in acc_dict.items()
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
        accept_dicts = [{}]
        # should not be unbounded
        check_length = 0
        if joint_iteration is not None:
            for i, key in enumerate(joint_iteration):
                if i == 0:
                    check_length = len(joint_dict[key])
                    accept_dicts = [{} for i in range(check_length)]
                assert check_length == len(joint_dict[key])
                for acc_dict, val in zip(accept_dicts, joint_dict[key]):
                    acc_dict[key] = val
        return accept_dicts

    def skip_config(self, config_settings: Dict) -> bool:
        """This function returns true if no execution with this config should be executed"""
        return False

    def launch_runs(self):
        """Execute all runs specified in init."""
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
                subprocess.run(launch_command, shell=True, check=True)

    @staticmethod
    def dict_to_arg(
        in_dict: Dict[str, Any], prefix: str = "", key_to_arg: str = "="
    ) -> str:
        """Convert a dictionary to a string of command-line arguments.

        Args:
            dict (dict): A dictionary containing key-value pairs.
            prefix (str, optional): A prefix to be added before each key. Defaults to "".
            key_to_arg (str, optional): The delimiter between key and value. Defaults to "=".

        Returns:
            str: A string of command-line arguments.

        Example:
        given `in_dict = {"arg1": "value1", "arg2": "value2"}` and `prefix = "--"`,
        the function returns `"--arg1=value1 --arg2=value2"`.
        """
        argument_string = ""
        for key, value in in_dict.items():
            argument_string += f"{prefix}{key}{key_to_arg}{value} "
        return argument_string

    @staticmethod
    def finalize_paths(paths: Union[str, List[str]], on_cluster: bool) -> List[str]:
        """Finalize relative paths according to the environment.

        Args:
            paths (Union[str, List[str]]): A relative path or a list of relative paths to be finalized.
            on_cluster (bool): A flag indicating whether the code is on a cluster.

        Returns:
            List[str]: A list of paths where relative paths are finalized.
        """
        if on_cluster:
            log_path = CLUSTER_LOG_PATH
        else:
            log_path = LOCAL_LOG_PATH
        if not isinstance(paths, (list, tuple)):
            paths = [paths]

        out = []
        for path in paths:
            out.append(os.path.join(log_path, path))
        return out


class ExperimentLauncher(BaseLauncher):
    def skip_config(self, config_settings: Dict) -> bool:
        """This function returns true if no execution with this config should be executed.
        Is true if values for keys in config_settings are:
        1. sem_sl.eman is True and model.use_ema is False
        2. model.finetune is True and model.freeze_encoder is True
        3. query in [bald, batchbald] and model.dropout_p=0


        Args:
            config_settings (Dict): dict for execution of singular experiment

        Returns:
            bool: True if conditions are met, else False
        """
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

    """Returns the value of a predefined config in hydra, given the key etc."""

    @staticmethod
    def access_config_value(
        key: str,
        config_name: str,
        config_dict: Dict[str, Union[str, List[str]]],
        hparam_dict: Dict[str, Any],
        default_struct: bool = True,
        file_ending: str = ".yaml",
    ) -> Any:
        """
        Accesses a configuration value based on the provided key, configuration name,
        and additional parameters.

        Args:
            key (str): The key for the configuration value.
            config_name (str): The name of the configuration.
            config_dict (Dict[str, Union[str, List[str]]]): A dictionary containing
                configuration information. The value associated with the configuration
                name can be a single string or a list of strings (e.g., for multiple files).
            hparam_dict (Dict[str, Any]): A dictionary containing hyperparameter information.
            default_struct (bool, optional): Flag indicating whether to use default
                folder structure for configuration files. Defaults to True.
            file_ending (str, optional): The file ending for configuration files.
                Defaults to ".yaml".

        Returns:
            Any: The value associated with the specified key in the configuration.

        Raises:
            NotImplementedError: If the specified configuration loading method is not implemented.
            FileNotFoundError: If the specified configuration file is not found.
            KeyError: If the specified key is not found in the configuration file.

        Note:
            The function first checks if the key is present in the provided
            hyperparameter dictionary. If found, it returns the corresponding value.
            If not, and `default_struct` is set to True, it attempts to load the
            configuration file based on the specified configuration name and key.
            The file is assumed to be located in the default folder structure under
            "src/config/{config_name}/{filename + file_ending}". If successful,
            it returns the value associated with the key in the configuration file.
            If no configuration loading method is successful, an error is raised.
        """
        full_key = f"{config_name}.{key}"

        if full_key in hparam_dict:
            return hparam_dict[full_key]

        if default_struct:
            current_dir = os.path.dirname(os.path.realpath(__file__))
            base_dir = os.path.join(
                current_dir, ".."
            )  # Assuming script is in "src" directory

            filenames = config_dict.get(config_name)
            if not filenames:
                raise ValueError(
                    f"No filenames specified for config_name '{config_name}'"
                )

            if not isinstance(filenames, list):
                filenames = [filenames]

            if len(filenames) > 1:
                raise NotImplementedError(
                    "Currently only one file parameter can be loaded in this fashion."
                )

            filename = filenames[0]
            path_to_config = os.path.join(
                base_dir, "config", config_name, filename + file_ending
            )

            try:
                with open(path_to_config, "r") as stream:
                    value = yaml.safe_load(stream)[key]
                return value
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Configuration file not found: {path_to_config}"
                )
            except KeyError:
                raise KeyError(f"Key not found in configuration file: {key}")

        raise NotImplementedError(
            "Custom configuration loading methods are not implemented."
        )

    @staticmethod
    def modify_params_for_args(
        launcher_args: Namespace, config_dict: Dict, hparam_dict: Dict
    ):
        """Modifies configuration and hyperparameter dictionaries based on the provided
        command-line arguments.

        Args:
            launcher_args (Namespace): The parsed command-line arguments.
            config_dict (Dict): A dictionary containing configuration information.
            hparam_dict (Dict): A dictionary containing hyperparameter information.

        Returns:
            Tuple[Dict, Dict]: Modified configuration and hyperparameter dictionaries.

        Note:
            This function is a part of the `BaseLauncher` class and delegates the core
            modification logic to the base class's `modify_params_for_args` method.
            After the base modification, it checks if a specific load architecture key
            is present in the hyperparameter dictionary. If so, it further modifies the
            hyperparameters based on the loaded architecture information.

            If the 'model.load_pretrained' key is present and set to True, it retrieves
            the model and dataset names from the configuration dictionaries and uses them
            to generate paths to pretrained model checkpoints. These paths are then
            associated with the 'model.load_pretrained' key in the hyperparameter dictionary.

            Finally, the generated paths are finalized using `ExperimentLauncher.finalize_paths`
            based on the provided command-line arguments.

            The modified configuration and hyperparameter dictionaries are returned.
        """
        config_dict, hparam_dict = BaseLauncher.modify_params_for_args(
            launcher_args, config_dict, hparam_dict
        )
        load_arch = "model.load_pretrained"
        if load_arch in hparam_dict:
            if hparam_dict[load_arch] is None:
                hparam_dict["model.load_pretrained"] = "Null"
            else:
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
                hparam_dict[
                    "model.load_pretrained"
                ] = ExperimentLauncher.finalize_paths(
                    hparam_dict["model.load_pretrained"],
                    on_cluster=launcher_args.bsub,
                )

        return config_dict, hparam_dict

    @staticmethod
    def dict_to_arg(dict: Dict[str, Any], prefix: str = "", key_to_arg: str = "="):
        """
        Converts a dictionary into a string of command-line arguments.

        Args:
            dictionary (Dict[str, Any]): The dictionary to convert into command-line arguments.
            prefix (str, optional): The prefix to prepend to each dictionary key in the argument string.
                Defaults to an empty string.
            key_to_arg (str, optional): The string used to separate dictionary keys and values in the argument string.
                Defaults to "=".

        Returns:
            str: A string containing command-line arguments based on the provided dictionary.

        Note:
            This function iterates through the key-value pairs in the input dictionary and
            constructs a string where each key-value pair is formatted as "{prefix}{key}{key_to_arg}{value}".
            If the value is `None`, it is represented as "Null" in the argument string.

        Example:
            If the input dictionary is {'param1': 42, 'param2': 'value', 'param3': None}, and the
            prefix is '--', and the key_to_arg is '=', the resulting argument string will be:
            "--param1=42 --param2=value --param3=Null"
        """
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
        load_pretrained, on_cluster=launcher_args.bsub
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
