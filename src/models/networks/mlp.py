import torch
import torch.nn as nn
from utils.consistent_mc_dropout import ConsistentMCDropout


class MLP(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        hidden_dims=[20],
        non_lin=torch.nn.ReLU,
        bn=False,
        dropout_p=0,
    ):
        """Defines a Multilayer Perceptron.
        Ordering of interlayer fct: Linear -> BN -> Non-Lin -> Dropout.
        This scheme was taken from:
        Understanding the Disharmony between Dropout and Batch Normalization by Variance Shift (CVPR2019)
        https://arxiv.org/abs/1801.05134"""
        super(MLP, self).__init__()
        layers = [nn.Linear(dim_in, hidden_dims[0])]
        for i in range(len(hidden_dims) - 1):
            if bn:
                layers.append(nn.BatchNorm1d(hidden_dims[i]))
            layers.append(non_lin())
            if dropout_p > 0:
                layers.append(ConsistentMCDropout(p=dropout_p))
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        if bn:
            layers.append(nn.BatchNorm1d(hidden_dims[-1]))
        if dropout_p > 0:
            layers.append(ConsistentMCDropout(p=dropout_p))
        layers.append(non_lin())
        layers.append(nn.Linear(hidden_dims[-1], dim_out))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        z = self.layers(x)
        return z
