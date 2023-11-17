from torch import Tensor

from models.bayesian_module import BayesianModule

from .mlp import MLP
from .registry import register_model


class BayesianMLP(BayesianModule):
    def __init__(
        self, num_classes, num_features, hidden_dims=(20, 20), dropout_p=0, bn=True
    ):
        super().__init__()
        self.mlp = MLP(
            dim_in=num_features,
            dim_out=num_classes,
            hidden_dims=hidden_dims,
            dropout_p=dropout_p,
            bn=bn,  # all experiments prior to 2022-04-12 have batchnorm as False!
        )

    def get_features(self, x):
        return x

    def mc_forward_impl(self, input: Tensor):
        input = self.mlp(input)

        return input


# TODO: Generalize this
@register_model
def get_cls_model(
    config, num_classes: int = 10, data_shape=(2), **kwargs
) -> BayesianMLP:
    if len(data_shape) != 1:
        raise Exception("This Model is not compatible with this input shape")
    num_features = data_shape[0]
    dropout_p = config.model.dropout_p
    hidden_dims = config.model.hidden_dims
    bn = config.model.use_bn
    return BayesianMLP(
        num_classes, num_features, hidden_dims=hidden_dims, dropout_p=dropout_p, bn=bn
    )
