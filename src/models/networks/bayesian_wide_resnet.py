import torch
import torch.nn as nn

from models.bayesian_module import BayesianModule, ConsistentMCDropout

from .mlp import MLP
from .registry import register_model
from .wide_resnet import Wide_ResNet, build_wideresnet


class BayesianWideResNet(BayesianModule):
    def __init__(
        self,
        depth: int,
        widen_factor: int,
        dropout_p: float,
        num_classes: int,
        small_head: bool = True,
    ):
        super().__init__()
        self.resnet = build_wideresnet(
            depth=depth, widen_factor=widen_factor, dropout=0, num_classes=10
        )
        self.z_dim = self.resnet.linear.in_features
        self.resnet.linear = nn.Identity()
        if num_classes != 0:
            if small_head:
                self.classifier = nn.Sequential(
                    ConsistentMCDropout(p=dropout_p), nn.Linear(self.z_dim, num_classes)
                )
            else:
                self.classifier = nn.Sequential(
                    ConsistentMCDropout(p=dropout_p),
                    MLP(self.z_dim, num_classes, hidden_dims=[self.z_dim], bn=True),
                )
        else:
            self.classifier = nn.Identity()

    def det_forward_impl(self, input: torch.Tensor) -> torch.Tensor:
        return self.resnet(input)

    def mc_forward_impl(self, mc_input_BK: torch.Tensor) -> torch.Tensor:
        return self.classifier(mc_input_BK)

    def get_features(self, x):
        out = self.resnet(x)
        return out


def build_bayesian_wide_resnet(
    depth: int,
    widen_factor: int,
    dropout_p: float,
    num_classes: int,
    small_head: bool = False,
):
    return BayesianWideResNet(depth, widen_factor, dropout_p, num_classes, small_head)


@register_model
def get_cls_model(
    config, num_classes: int = 10, data_shape=[32, 32, 3], **kwargs
) -> Wide_ResNet:
    if len(data_shape) != 3:
        raise Exception("This Model is not compatible with this input shape")
    if data_shape[2] != 3:
        raise Exception("This Model only works for image data with 3 channels")
    channels_in = data_shape[2]
    small_head = config.model.small_head
    # dropout_p = config.model.dropout_p
    return build_bayesian_wide_resnet(
        depth=config.model.model_depth,
        widen_factor=config.model.model_width,
        dropout_p=config.model.dropout_p,
        num_classes=num_classes,
        small_head=small_head,
    )
