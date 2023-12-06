from typing import Any, Dict, Iterator, List, Tuple

import torch


def exclude_from_wt_decay(
    named_params: Iterator[Tuple[str, torch.Tensor]],
    weight_decay: float,
    skip_list: List[str] = ["bias", "bn"],
    learning_rate: float = None,
) -> List[Dict[str, Any]]:
    """Exclude parameters from weight decay and get groups with specific instructions for optimizers.
    Ignores parameters where requires_grad is False.

    Args:
        named_params (Iterator[Tuple[str, torch.Tensor]]): List of named parameters
        weight_decay (float): weight decay param for optimizer
        skip_list (List[str], optional): Parameters without weight decay. Defaults to ["bias", "bn"].
        learning_rate (float, optional): _description_. Defaults to None.

    Returns:
        List[dict[str, Any]]: [Paramdict with weight decay, Paramdict without weight decay]
    """
    params = []
    excluded_params = []

    for name, param in named_params:
        if not param.requires_grad:
            continue
        elif any(layer_name in name for layer_name in skip_list):
            excluded_params.append(param)
        else:
            params.append(param)
    params = [
        {"params": params, "weight_decay": weight_decay},
        {"params": excluded_params, "weight_decay": 0},
    ]
    if learning_rate is not None:
        for param_group in params:
            param_group["lr"] = learning_rate
    return params


def load_from_ssl_checkpoint(model: torch.nn.Module, path: str):
    """Loads the parameters from path"""
    # I know, this is ugly but for now me and you will have to leave with it...
    # at least until I find time to fix this. (or you :))

    ### Impelmentation ###
    # add loading of alternatie ssl checkpoints here.

    # Approach: load only the state dict and then take the values for the encoder.
    #################### SUPER UGLY CODE STARTS HERE ##################
    # Change this part so that there are no cross dependencies between skripts and source folder!
    import os
    import sys

    add_to_sys_folder = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "ssl",
    )

    sys.path.append(add_to_sys_folder)
    from train_simclr import SimCLR_algo as SimCLR

    ################### SUPER UGLY CODE ENDS HERE ######################
    # Deprecated due to WideResNet
    # from pl_bolts.models.self_supervised.simclr.simclr_module import SimCLR

    model_ssl = SimCLR.load_from_checkpoint(path, map_location="cpu", strict=False)
    param_names = model.resnet.state_dict().keys()
    load_dict = dict([(n, model_ssl.encoder.state_dict()[n]) for n in param_names])
    skip_names = [
        n for n in model_ssl.encoder.state_dict().keys() if n not in param_names
    ]
    msg = model.resnet.load_state_dict(load_dict)
    print(msg)
    print(f"Skipped Parameters: {skip_names}")


def freeze_layers(
    model: torch.nn.Module, freeze_string: str = None, keep_string: str = None
) -> None:
    """Freezes specific parameters in model if name is matching freeze_string but not keep_string.

    Args:
        model (torch.nn.Module): model, changed in place.
        freeze_string (str, optional): string which is matched with parameter name. Defaults to None.
        keep_string (str, optional): string which is matched with parameter name. Defaults to None.
    """
    for param in model.named_parameters():
        if freeze_string is None and keep_string is None:
            param[1].requires_grad = False
        if freeze_string is not None and freeze_string in param[0]:
            param[1].requires_grad = False
        if keep_string is not None and keep_string not in param[0]:
            param[1].requires_grad = False


def unfreeze_layers(model: torch.nn.Module, unfreeze_string: str = None) -> None:
    """Unfreezes specific parameters in model if name is matching unfreeze_string.

    Args:
        model (torch.nn.Module): model, changed in place.
        unfreeze_string (str, optional): string which is matched with parameter name. Defaults to None.
    """
    for param in model.named_parameters():
        if unfreeze_string is None or unfreeze_string in param[0]:
            param[1].requires_grad = True
