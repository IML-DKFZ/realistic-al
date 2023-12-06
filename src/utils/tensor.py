from typing import Any

import numpy as np
import torch


def to_numpy(data: Any) -> np.ndarray:
    """Change data carrier data to a numpy array

    Args:
        data (Any): data carrier (torch, list, tuple, numpy)

    Returns:
        np.ndarray: array carrying data from data
    """
    if torch.is_tensor(data):
        return data.to("cpu").detach().numpy()
    elif isinstance(data, np.ndarray):
        return data
    elif isinstance(data, (tuple, list)):
        return np.array(data)
    # this might lead to bugs, however it should catch numeric data.
    elif hasattr(data, "__len__") is False:
        return np.array([data])
    elif isinstance(data, (float, int)):
        return np.array([data])
    else:
        raise TypeError(
            "Object data of type {} cannot be converted to np.ndarray".format(data.type)
        )
