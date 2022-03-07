import torch


def exclude_from_wt_decay(
    named_params, weight_decay, skip_list=["bias", "bn"], learning_rate=None
):
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
    # TODO: Generalize this function!!!!
    #################### SUPER UGLY CODE STARTS HERE ##################
    # Change this part so that there are no cross dependencies between skripts and source folder!
    import os
    import sys

    add_to_sys_folder = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "skripts",
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


def freeze_layers(model, freeze_string=None, keep_string=None):
    for param in model.named_parameters():
        if freeze_string is None and keep_string is None:
            param[1].requires_grad = False
        if freeze_string is not None and freeze_string in param[0]:
            param[1].requires_grad = False
        if keep_string is not None and keep_string not in param[0]:
            param[1].requires_grad = False


def unfreeze_layers(model, unfreeze_string=None):
    for param in model.named_parameters():
        if unfreeze_string is None or unfreeze_string in param[0]:
            param[1].requires_grad = True
