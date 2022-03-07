import json
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
    if isinstance(path, str):
        path = Path(path)
    if not (".json" == path.suffix):
        path = Path(str(path) + ".json")

    with open(path, "w") as f:
        json.dump(data, f, indent=indent, **kwargs)
