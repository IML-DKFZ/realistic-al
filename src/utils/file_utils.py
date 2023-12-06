from collections import MutableMapping
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import entropy

from utils import io
from utils.tensor import to_numpy


def get_all_files_naming(root: Path, pattern: str) -> List[Path]:
    """Returns all files in the root folder matching pattern in a sorted order.

    Args:
        root (Path): Root for search of pattern
        pattern (str): pattern file should match

    Returns:
        List[Path]: Files in subdir matching pattern
    """
    files = []

    files = []
    for file in root.rglob(pattern):
        files.append(file)
    files.sort()
    return files


from collections import MutableMapping


# code to convert init_dict to flattened dictionary
# default separator '_'
def convert_flatten(
    d: MutableMapping, parent_key: str = "", sep: str = "."
) -> Dict[str, Any]:
    """Recursively go through nested mutable mapping and return a flattened version with keys being appended and seperated by sep.

    Args:
        d (MutableMapping): nested dictionary to be read out.
        parent_key (str, optional): key used as a prefix. Defaults to "".
        sep (str, optional): seperator between layers. Defaults to ".".

    Returns:
        Dict[str, Any]: Flattened Dictonary
    """
    items = []
    for k in d.keys():
        new_key = parent_key + sep + k if parent_key else k
        try:
            v = d[k]
        except:
            v = "NaN"

        if isinstance(v, MutableMapping):
            items.extend(convert_flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))

    return dict(items)


def get_experiment_configs_df(
    experiment_path: Path, name: str = None, pattern: str = "config.yaml"
) -> pd.DataFrame:
    """Load configs for experiments and return them as a Dataframe.

    Args:
        experiment_path (Path): base path from which experiments are searched with rglob.
        name (str, optional): value to be added under key "Name", if none use name of path. Defaults to None.
        pattern (str, optional): files that are searched for as configs. Defaults to "config.yaml".

    Returns:
        pd.DataFrame: 2D tabular data with all keys from config.
    """
    files = get_all_files_naming(experiment_path, pattern)
    out_dicts = []
    print("Loading Experiment:", experiment_path)
    for file in files:
        print("Loading File:", file)
        loaded = io.load_omega_conf(file)
        flattened = convert_flatten(loaded)
        out_dicts.append(flattened)

    dataframe = []
    for i, out_dict in enumerate(out_dicts):
        df_temp = pd.DataFrame(out_dict)
        df_temp["version"] = i
        dataframe.append(df_temp)
    dataframe = pd.concat(dataframe)
    if name is None:
        name = experiment_path.name
    dataframe["Name"] = name
    return dataframe


def get_experiment_df(
    experiment_path: Path,
    name: str = None,
    pattern: str = "stored.npz",
    verbose: bool = False,
) -> pd.DataFrame:
    """Returns a cleaned dataframe with results from an experiment with a npz save structure.

    Args:
        experiment_path (Path): base path from which rglob search for pattern is executed.
        name (str, optional): value to be added under key "Name", if none use name of path. Defaults to None.
        pattern (str, optional): files that are searched for as results. Defaults to "stored.npz".
        verbose (bool, optional): function talks to you. Defaults to False.

    Returns:
        pd.DataFrame: contains multiple experiments differentiated by version.
    """
    files = get_all_files_naming(experiment_path, pattern)
    out_dicts = []
    if len(files) == 0:
        return None
    for file in files:
        if verbose:
            print("\tLoading File:", file)
        filetype = pattern.split(".")[-1]
        if filetype == "npz":
            loaded = np.load(file)
            data_dict = dict()
            for key, val in loaded.items():
                data_dict[key] = val
        elif filetype == "csv":
            csv_dict = pd.read_csv(file, index_col=0)
            data_dict = csv_dict.to_dict(orient="list")
        out_dicts.append(data_dict)

    dataframe = []
    for i, out_dict in enumerate(out_dicts):
        # throw out all data which cannot be easily converted to a dataframe
        if len(out_dict) == 0:
            if verbose:
                print("File {} \nIs empty and will be skipped!".format(file))
            return None
        for key in out_dict:
            pop_keys = []
            out_dict[key] = to_numpy(out_dict[key])
            if len(out_dict[key].shape) > 1:
                pop_keys.append(key)

        ################################
        # Compute Entropies

        if "added_labels" in out_dict:
            unique = np.unique(out_dict["added_labels"])
            count_list = []
            for i, labels in enumerate(out_dict["added_labels"]):
                counts = np.bincount(labels, minlength=unique.max() + 1)
                count_list.append(counts)
            count_list = np.stack(count_list, axis=0)

            prob = count_list / np.sum(count_list, axis=1, keepdims=True)
            out_dict["Acquisition Entropy"] = entropy(prob, axis=1)
            count_list = np.cumsum(count_list, axis=0)
            prob = count_list / np.sum(count_list, axis=1, keepdims=True)
            out_dict["Dataset Entropy"] = entropy(prob, axis=1)
            out_dict["Dataset Entropy"][1:] = out_dict["Dataset Entropy"][:-1]
            out_dict["Dataset Entropy"][0] = np.NaN

        ################################

        for key in pop_keys:
            out_dict.pop(key)

        # this part of the code is quite ugly but was necessary.
        pop_keys = []
        for key in out_dict:
            if len(out_dict[key].shape) == 2:
                if out_dict[key].shape[1] == 1:
                    out_dict[key] = out_dict[key].squeeze(1)
                else:
                    pop_keys.append(key)
        for key in pop_keys:
            out_dict.pop(key)

        df_temp = pd.DataFrame(out_dict)
        df_temp["version"] = i
        dataframe.append(df_temp)
    dataframe = pd.concat(dataframe)
    if name is None:
        name = experiment_path.name
    dataframe["Name"] = name
    return dataframe


def get_nested_file_list(
    root_path: Path, pardir: str, subfolder: str
) -> List[List[Path]]:
    """Scrape the level below for all folders matching name pardir,
    go into subpath and get paths to all files.

    Useful for:
    root_path/*pardir/subpath/files

    Args:
        experiment_path (Path): from where to look
        pardir (str): pattern to match
        subfolder (str): directory to go in

    Returns:
        List[List[Path]]: files in lists according to *pardir
    """
    files = list()
    for path in root_path.iterdir():
        if path.is_dir() and pardir in path.name:
            subpath = path / subfolder
            if subpath.is_dir():
                path_files = list()
                for file in subpath.iterdir():
                    path_files.append(file)
                if len(path_files) > 0:
                    files.append(path_files)
    files.sort(key=lambda x: x[0])
    return files
