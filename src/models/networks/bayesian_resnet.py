import torch
import torch.nn as nn
import torchvision.models as models
from loguru import logger

from models.bayesian_module import BayesianModule, ConsistentMCDropout

from .mlp import MLP
from .registry import register_model


class ResNet(BayesianModule):
    def __init__(
        self,
        base_model="resnet50",
        cifar_stem=True,
        channels_in=3,
        num_classes=0,
        dropout_p=0.5,
        small_head=True,
        weights=None,
    ):
        """obtains the ResNet for use as an Encoder, with the last fc layer
        exchanged for an identity

        Args:
            base_model (str, optional): [description]. Defaults to "resnet50".
            cifar_stem (bool, optional): [input resolution of 32x32]. Defaults to True.
            channels_in (int, optional): [description]. Defaults to 3.
            num_classes (int, optional): number of classes (if = 0) purely feature extractor
        """
        super().__init__()
        self.resnet_dict = {
            "resnet18": models.resnet18(weights=weights),
            "resnet50": models.resnet50(weights=weights),
        }

        self.resnet = self._get_basemodel(base_model)
        num_ftrs = self.resnet.fc.in_features

        # change 1st convolution to work with inputs [channels_in x 32 x 32]
        if cifar_stem:
            conv1 = nn.Conv2d(
                channels_in, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            nn.init.kaiming_normal_(conv1.weight, mode="fan_out", nonlinearity="relu")
            self.resnet.conv1 = conv1
            self.resnet.maxpool = nn.Identity()
        elif channels_in != 3:
            conv1 = nn.Conv2d(
                channels_in, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            nn.init.kaiming_normal_(conv1.weight, mode="fan_out", nonlinearity="relu")
            self.resnet.conv1 = conv1
        self.z_dim = num_ftrs

        self.resnet.fc = nn.Identity()

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

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            logger.info("Feature extractor: {}".format(model_name))
            return model
        except:
            raise (
                "Invalid model name. Check the config file and pass one of: resnet18 or resnet50"
            )


@register_model
def get_cls_model(
    config,
    base_model="resnet18",
    num_classes: int = 10,
    data_shape=[32, 32, 3],
    **kwargs
) -> ResNet:
    if len(data_shape) != 3:
        raise Exception("This Model is not compatible with this input shape")
    cifar_stem = False
    if data_shape[0] == 32 and data_shape[1] == 32:
        cifar_stem = True
    channels_in = data_shape[2]
    dropout_p = config.model.dropout_p
    small_head = config.model.small_head
    try:
        weights = config.model.weights
    except:
        weights = None
    base_model = config.model.base_model

    return ResNet(
        base_model=base_model,
        cifar_stem=cifar_stem,
        channels_in=channels_in,
        num_classes=num_classes,
        dropout_p=dropout_p,
        small_head=small_head,
        weights=weights,
    )
