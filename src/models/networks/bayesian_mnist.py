import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils import data

from models.bayesian_module import (
    BayesianModule,
    ConsistenMCDropout2D,
    ConsistentMCDropout,
)

from .registry import register_model


# MNIST BayesianNet from BatchBALD
class BayesianNet(BayesianModule):
    def __init__(self, num_classes, num_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=5)
        self.conv1_drop = ConsistenMCDropout2D()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv2_drop = ConsistenMCDropout2D()
        self.fc1 = nn.Linear(1024, 128)
        self.fc1_drop = ConsistentMCDropout()
        self.fc2 = nn.Linear(128, num_classes)

    def mc_forward_impl(self, input: Tensor):
        input = F.relu(F.max_pool2d(self.conv1_drop(self.conv1(input)), 2))
        input = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(input)), 2))
        input = input.view(-1, 1024)
        input = F.relu(self.fc1_drop(self.fc1(input)))
        input = self.fc2(input)
        # input = F.log_softmax(input, dim=1)

        return input


@register_model
def get_cls_model(
    config, num_classes: int = 10, data_shape=[28, 28, 1], **kwargs
) -> BayesianNet:
    if len(data_shape) != 3:
        raise Exception("This Model is not compatible with this input shape")
    if data_shape[0] != 28 or data_shape[1] != 28:
        raise Exception(
            "This Model is not compatible with this input shape {}".format(data_shape)
        )
    num_channels = data_shape[2]
    return BayesianNet(num_classes, num_channels=num_channels)
