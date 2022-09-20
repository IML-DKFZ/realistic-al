import enum
import os
from pathlib import Path
from typing import List, Iterable, Any, Optional

import numpy as np
import pandas as pd
from utils.tensor import to_numpy
from utils import io
from scipy.stats import entropy


def get_all_files_naming(root: Path, pattern: str) -> List[Path]:
    """Returns all files in the root folder matching pattern in a sorted order

    Args:
        root (Path): Root for search of pattern
        pattern (str): pattern file should match

    Returns:
        List[Path]: Files in subdir matching pattern
    """
    files = []
    # for root_dirlist, dirlist, filelist in os.walk(root):
    #     for file in filelist:
    #         if file == pattern:
    #             print(root_dirlist)
    #             print(dirlist)
    #             print(filelist)
    #             # if re.match(pattern, file):
    #             # files.append(Path(root_dir) / dir / file)

    # this works only for 1 level!
    files = []
    for file in root.rglob(pattern):
        files.append(file)
    files.sort()
    return files


from collections import MutableMapping

# code to convert ini_dict to flattened dictionary
# default separator '_'
def convert_flatten(d: MutableMapping, parent_key="", sep="."):
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
    experiment_path: Path, name: Optional[str] = None, pattern: str = "config.yaml"
):
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
        df_temp["version"] = i  # ToDo - change this to version
        dataframe.append(df_temp)
    dataframe = pd.concat(dataframe)
    if name is None:
        name = experiment_path.name
    dataframe["Name"] = name
    return dataframe


def get_experiment_df(
    experiment_path: Path,
    name: Optional[str] = None,
    pattern: str = "stored.npz",
    verbose: bool = False,
) -> pd.DataFrame:
    """Returns a cleaned dataframe with results from an experiment with a npz save structure.

    Args:
        experiment_path (Path): _description_
        name (Optional[str], optional): _description_. Defaults to None.
        pattern (str, optional): _description_. Defaults to "stored.npz".

    Returns:
        _type_: _description_
    """
    files = get_all_files_naming(experiment_path, pattern)
    out_dicts = []
    print("Loading Experiment:", experiment_path)
    print(f"Found num files: {len(files)}")
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

    # TODO: possible extension, add config values from hydra (hparams.yaml)
    dataframe = []
    for i, out_dict in enumerate(out_dicts):
        # throw out all data which cannot be easily converted to a dataframe
        if len(out_dict) == 0:
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

        # TODO: Check if this is needed generally!
        for key in out_dict:
            if len(out_dict[key].shape) == 2:
                out_dict[key] = out_dict[key].squeeze(1)

        df_temp = pd.DataFrame(out_dict)
        df_temp["version"] = i  # ToDo - change this to version
        dataframe.append(df_temp)
    dataframe = pd.concat(dataframe)
    if name is None:
        name = experiment_path.name
    dataframe["Name"] = name
    return dataframe


def get_experiment_dicts(experiment_path: Path) -> List[dict]:
    file_paths = get_nested_file_list(
        experiment_path, pardir="loop", subfolder="save_dict"
    )
    dictlist = [load_files_to_dict(pathfiles) for pathfiles in file_paths]
    return dictlist


def load_files_to_dict(files: Iterable[Path]):
    def to_dict(object: Any):
        out_dict = dict()
        for key in object.keys():
            out_dict[key] = object[key]
        return out_dict

    out_dict = dict()
    for file in files:
        key = file.with_suffix("").name
        data = to_dict(np.load(file))
        out_dict[key] = data
    return out_dict


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
