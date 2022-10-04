import json
import os
import pickle
from omegaconf import DictConfig, OmegaConf
import yaml
from typing import Any, Union
from pathlib import Path


def save_json(data: Any, path: Union[Path, str], indent: int = 4, **kwargs):
    """
    Load json file
    Args:
        data: data to save to json
        path: path to json file
        indent: passed to json.dump
        **kwargs: keyword arguments passed to :func:`json.dump`
    """
    suffix = ".json"
    path = prepare_path(path, suffix)

    with open(path, "w") as f:
        json.dump(data, f, indent=indent, **kwargs)


def load_json(path: Union[Path, str], **kwargs) -> Any:
    suffix = ".json"
    path = prepare_path(path, suffix)

    with open(path, "r") as f:
        data = json.load(f, **kwargs)
    return data


def save_pickle(data: Any, path: Union[Path, str], **kwargs):
    suffix = ".pkl"
    path = prepare_path(path, suffix)

    with open(path, "wb") as f:
        pickle.dump(data, f, **kwargs)


def load_pickle(path: Union[Path, str], **kwargs) -> Any:
    suffix = ".pkl"
    path = prepare_path(path, suffix)

    with open(path, "rb") as f:
        data = pickle.load(f, **kwargs)
    return data


def save_yaml(data: Any, path: Union[Path, str], **kwargs):
    suffix = ".yaml"
    path = prepare_path(path, suffix)

    with open(path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, **kwargs)


def load_yaml(path: Union[Path, str], **kwargs) -> Any:
    suffix = ".yaml"
    path = prepare_path(path, suffix)

    with open(path, "r") as f:
        data = yaml.save_load(f, **kwargs)
    return data


def save_omega_conf(
    cfg: OmegaConf,
    path: Union[Path, str],
    suffix: str = ".yaml",
    resolve: bool = False,
    **kwargs
):
    if suffix not in [".yaml", ".pkl"]:
        raise TypeError("File can only be saved as .pkl or .yaml")
    path = prepare_path(path, suffix)
    with open(path, "w") as f:
        # if resolve is set to true, then all values which are ${value} are replaced with is value!
        OmegaConf.save(cfg, f, resolve=resolve)


def load_omega_conf(
    path: Union[Path, str], overwrite_values: bool = True
) -> DictConfig:
    path = Path(path)
    if path.suffix not in [".yaml", ".pkl"]:
        raise TypeError("File can only be loaded from .pkl or .yaml")
    with open(path, "r") as f:
        cfg = OmegaConf.load(f)
    if overwrite_values:
        cfg.trainer.data_root = os.getenv("DATA_ROOT")
        cfg.model.load_pretrained = None
    return cfg


def write_file(data: str, path: Union[Path, str], suffix: str = ".txt", **kwargs):
    path = prepare_path(path, suffix)

    with open(path, "w") as f:
        f.writelines(data, f, default_flow_style=False, **kwargs)


def prepare_path(path: Union[Path, str], suffix: str) -> Path:
    if isinstance(path, str):
        path = Path(path)
    if not (suffix == path.suffix):
        path = Path(str(path) + suffix)
    return path
