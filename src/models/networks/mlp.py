import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self, dim_in, dim_out, hidden_dims=[20], non_lin=torch.nn.ReLU, bn=False
    ):
        super(MLP, self).__init__()
        layers = [nn.Linear(dim_in, hidden_dims[0])]
        for i in range(len(hidden_dims) - 1):
            if bn:
                layers.append(nn.BatchNorm1d(hidden_dims[i]))
            layers.append(non_lin())
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        if bn:
            layers.append(nn.BatchNorm1d(hidden_dims[-1]))
        layers.append(non_lin())
        layers.append(nn.Linear(hidden_dims[-1], dim_out))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        z = self.layers(x)
        return z
