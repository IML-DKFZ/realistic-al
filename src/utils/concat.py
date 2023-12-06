from abc import abstractclassmethod
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .tensor import to_numpy


def get_batch_data(batch: Any, out_dict: dict = None) -> Dict[str, np.ndarray]:
    """Access data inside of a batch and return it in form of a dictionary.
    If no out_dict is given then, a new dict is created and returned.

    Args:
        batch (Any): Batch from a dataloader
        out_dict (dict, optional): dictionary for extension. Defaults to None.

    Returns:
        Dict[str, np.ndarray]: dictionary carrying batch data
    """
    if out_dict is None:
        out_dict = dict()
    if isinstance(batch, (list, tuple)):
        x, y = batch
    else:
        raise NotImplemented(
            "Currently this function is only implemented for batches of type tuple"
        )
    out_dict["data"] = x
    if y is not None:
        out_dict["label"] = y
    else:
        # create dummy label with same shape as predictions if no labels applicable
        # value of -1
        dummy_label = torch.ones(x.shape[0], dtype=torch.long) * -1
        out_dict["label"] = dummy_label
    for key, val in out_dict.items():
        out_dict[key] = to_numpy(val)
    return out_dict


# this function might also work based on an abstract class (see GetModelOutputs)
def concat_functional(
    chunked_data: Iterable,
    functions: Tuple[Callable[[Any], Dict[str, np.ndarray]]] = (get_batch_data,),
) -> Dict[str, np.ndarray]:
    """Concatenates outputs of functions from chunked_data.
    For a template regarding a function see `get_batch_data`.
    Also `AbstractBatchData` allows easy creation.

    Args:
        chunked_data (Iterable): chunked data for running model.
        functions (Tuple[Callable], optional): tuple of functions operating on data in chunked data. Defaults to (get_batch_data,).

    Returns:
        Dict[str, np.ndarray]: Dictionary contatining outputs for function on chunked_data
    """
    loader_dict = defaultdict(list)
    for data in chunked_data:
        batch_dict = {}
        for function in functions:
            batch_dict = function(data, out_dict=batch_dict)
        for key, val in batch_dict.items():
            loader_dict[key].append(val)
    # create new dictionary so as to keep loader_dict as list!
    out_loader_dict = dict()
    for key, value in loader_dict.items():
        out_loader_dict[key] = np.concatenate(value)
    return out_loader_dict


class AbstractBatchData(object):
    def __init__(self):
        """Abstract Class carrying utility to extract data from batches."""
        pass

    def __call__(self, batch, out_dict: dict = None):
        if out_dict is None:
            out_dict = dict()
        if isinstance(batch, (list, tuple)):
            x, y = batch
        else:
            raise NotImplemented(
                "Currently this function is only implemented for batches of type tuple"
            )
        out_dict = self.custom_call(x, out_dict=out_dict, batch=batch, y=y)
        for key, val in out_dict.items():
            out_dict[key] = to_numpy(val)
        return out_dict

    @abstractclassmethod
    def custom_call(self, x: Tensor, out_dict: dict, **kwargs):
        raise NotImplementedError


class GetClassifierOutputs(AbstractBatchData):
    def __init__(self, model: nn.Module, device="cuda:0"):
        """Class carrying logic to extract classifier outputs.
        Useful in combination with `concat_functional`.

        Args:
            model (nn.Module): Classifier returning log_probs
            device (str, optional): _description_. Defaults to "cuda:0".
        """
        super().__init__()
        self.model = model
        self.device = device
        self.model = self.model.to(device)

    def custom_call(self, x: Tensor, out_dict: dict, **kwargs):
        x = x.to(self.device)
        with torch.no_grad():
            out = self.model(x)
            out_dict["prob"] = out
            pred = torch.argmax(out, dim=-1)
            out_dict["pred"] = pred
        return out_dict
