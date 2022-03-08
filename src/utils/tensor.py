from typing import Any
import numpy as np
import torch


def to_numpy(data: Any) -> np.ndarray:
    """Change data carrier data to a numpy array

    Args:
        data (Any): data carrier (torch, list, tuple, numpy)

    Raises:
        TypeError: _description_

    Returns:
        np.ndarray: array carrying data from data
    """
    if torch.is_tensor(data):
        return data.to("cpu").detach().numpy()
    elif isinstance(data, (tuple, list)):
        return np.array(data)
    elif isinstance(data, np.ndarray):
        return data
    else:
        raise TypeError(
            "Object data of type {} cannot be converted to np.ndarray".format(data.type)
        )
