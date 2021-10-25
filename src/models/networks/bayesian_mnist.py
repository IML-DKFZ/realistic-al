import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from utils.consistent_mc_dropout import (
    ConsistenMCDropout2D,
    ConsistentMCDropout,
    BayesianModule,
)
from .registry import register_model

# MNIST BayesianNet from BatchBALD
class BayesianNet(BayesianModule):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
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

# TODO: Generalize this
@register_model
def get_cls_model(config, num_classes: int = 10, **kwargs) -> BayesianNet:
    return BayesianNet(num_classes)
