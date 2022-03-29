from pathlib import Path
from typing import List, Iterable, Any

import numpy as np


def get_experiment_dicts(experiment_path: Path):
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
