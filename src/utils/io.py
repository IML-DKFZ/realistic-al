import json
import os
import pickle
from pathlib import Path
from typing import Any, Union

import yaml
from omegaconf import DictConfig, OmegaConf


def save_json(data: Any, path: Union[Path, str], indent: int = 4, **kwargs) -> None:
    """Save json file.

    Args:
        data (Any): data to save to json
        path (Union[Path, str]): path to json file
        indent (int, optional): passed to json.dump. Defaults to 4.
        **kwargs: keyword arguments passed to :func:`json.dump`
    """
    suffix = ".json"
    path = prepare_path(path, suffix)

    with open(path, "w") as f:
        json.dump(data, f, indent=indent, **kwargs)


def load_json(path: Union[Path, str], **kwargs) -> Any:
    """Load json file.

    Args:
        path (Union[Path, str]): path to json file
        **kwargs: keyword arguments passed to :func:`json.load`

    Returns:
        Any: data of json file
    """
    suffix = ".json"
    path = prepare_path(path, suffix)

    with open(path, "r") as f:
        data = json.load(f, **kwargs)
    return data


def save_pickle(data: Any, path: Union[Path, str], **kwargs) -> None:
    """Save pickle file.

    Args:
        data (Any): data to save to pickle
        path (Union[Path, str]): path to pickle file
        **kwargs: keyword arguments passed to :func:`pickle.dump`
    """
    suffix = ".pkl"
    path = prepare_path(path, suffix)

    with open(path, "wb") as f:
        pickle.dump(data, f, **kwargs)


def load_pickle(path: Union[Path, str], **kwargs) -> Any:
    """Load pickle file.

    Args:
        path (Union[Path, str]): path to pickle file
        **kwargs: keyword arguments passed to :func:`pickle.load`

    Returns:
        Any: data of json file
    """
    suffix = ".pkl"
    path = prepare_path(path, suffix)

    with open(path, "rb") as f:
        data = pickle.load(f, **kwargs)
    return data


def save_yaml(data: Any, path: Union[Path, str], **kwargs) -> None:
    """Save yaml file.

    Args:
        data (Any): data to yaml to pickle
        path (Union[Path, str]): path to yaml file
        **kwargs: keyword arguments passed to :func:`yaml.safe_dump`
    """
    suffix = ".yaml"
    path = prepare_path(path, suffix)

    with open(path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, **kwargs)


def load_yaml(path: Union[Path, str], **kwargs) -> Any:
    """Load yaml file.

    Args:
        path (Union[Path, str]): path to yaml file

    Returns:
        Any: data of yaml file
    """
    suffix = ".yaml"
    path = prepare_path(path, suffix)

    with open(path, "r") as f:
        data = yaml.save_load(f, **kwargs)
    return data


def save_omega_conf(
    cfg: DictConfig,
    path: Union[Path, str],
    suffix: str = ".yaml",
    resolve: bool = False,
    **kwargs
) -> None:
    """Save nested dictconfig.

    Args:
        cfg (DictConfig): Config (possibly nested)
        path (Union[Path, str]): Path to save file
        suffix (str, optional): suffix for saved file in [.yaml, .pkl]. Defaults to ".yaml".
        resolve (bool, optional): if True, all values ${value} are replaced with read out value. Defaults to False.
    """
    if suffix not in [".yaml", ".pkl"]:
        raise ValueError("File can only be saved as .pkl or .yaml")
    path = prepare_path(path, suffix)
    with open(path, "w") as f:
        # if resolve is set to true, then all values which are ${value} are replaced with is value!
        OmegaConf.save(cfg, f, resolve=resolve)


def load_omega_conf(
    path: Union[Path, str], overwrite_values: bool = True
) -> DictConfig:
    """Load nested dictconfig from path.

    Args:
        path (Union[Path, str]): Path to saved file (needs to have suffix in [.yaml, .pkl])
        overwrite_values (bool, optional): Overwrites two values that can prevent loading of models. Defaults to True.


    Returns:
        DictConfig: data in omegaconf
    """
    path = Path(path)
    if path.suffix not in [".yaml", ".pkl"]:
        raise ValueError("File can only be loaded from .pkl or .yaml")
    with open(path, "r") as f:
        cfg = OmegaConf.load(f)
    if overwrite_values:
        cfg.trainer.data_root = os.getenv("DATA_ROOT")
        cfg.model.load_pretrained = None
    return cfg


def write_file(data: str, path: Union[Path, str], suffix: str = ".txt", **kwargs):
    """Write values to file in human readable format.

    Args:
        data (str): text to be written
        path (Union[Path, str]): path where file be written.
        suffix (str, optional): enforced suffix. Defaults to ".txt".
    """
    path = prepare_path(path, suffix)

    with open(path, "w") as f:
        f.writelines(data, f, default_flow_style=False, **kwargs)


def prepare_path(path: Union[Path, str], suffix: str) -> Path:
    """Add corresponding suffix to path if not already present.

    Args:
        path (Union[Path, str]): path to compare
        suffix (str): suffix for final file

    Returns:
        Path: path{.suffix}
    """
    if isinstance(path, str):
        path = Path(path)
    if not (suffix == path.suffix):
        path = Path(str(path) + suffix)
    return path
