import enum
import os
from pathlib import Path
from typing import List, Iterable, Any, Optional

import numpy as np
import pandas as pd
from utils.tensor import to_numpy


def get_all_files_naming(root: Path, pattern: str) -> List[Path]:
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


def get_experiment_df(
    experiment_path: Path, name: Optional[str] = None, pattern: str = "stored.npz"
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
    for file in files:
        print("Loading File:", file)
        loaded = np.load(file)
        data_dict = dict()
        for key, val in loaded.items():
            data_dict[key] = val
        out_dicts.append(data_dict)

    # TODO: possible extension, add config values from hydra (hparams.yaml)
    dataframe = []
    for i, out_dict in enumerate(out_dicts):
        # throw out all data which cannot be easily converted to a dataframe
        for key in out_dict:
            pop_keys = []
            out_dict[key] = to_numpy(out_dict[key])
            if len(out_dict[key].shape) > 1:
                pop_keys.append(key)
        for key in pop_keys:
            out_dict.pop(key)
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
        experiment_path, pardir="loop", subpath="save_dict"
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
    root_path: Path, pardir: str, subpath: str
) -> List[List[Path]]:
    """Scrape the level below for all folders matching name pardir,
    go into subpath and get paths to all files.

    Useful for:
    root_path/*pardir/subpath/files

    Args:
        experiment_path (Path): from where to look
        pardir (str): pattern to match
        subpath (str): directory to go in

    Returns:
        List[List[Path]]: files in lists according to *pardir
    """
    files = list()
    for path in root_path.iterdir():
        if path.is_dir() and pardir in path.name:
            subpath = path / subpath
            if subpath.is_dir():
                path_files = list()
                for file in subpath.iterdir():
                    path_files.append(file)
                if len(path_files) > 0:
                    files.append(path_files)
    files.sort(key=lambda x: x[0])
    return files
