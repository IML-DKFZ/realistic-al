import torch
import torch.nn as nn
import torch.nn.functional as F


class BayesianModule(nn.Module):
    k = None

    def __init__(self):
        """Bayesian Module which copies has a deterministic part for the forward pass and a stochastic one.
        Allows to compute deterministic part (CNN) first and then do the stochastic part afterward (e.g. classification head).
        """
        super().__init__()

    def forward(self, x: torch.Tensor, k: int):
        """Standard Forward Pass of the module.
        k is number of MC samples"""
        BayesianModule.k = k

        x = self.det_forward_impl(x)
        mc_input_BK = BayesianModule.mc_tensor(x, k)  # Bx .... --> B*kx ...
        mc_output_BK = self.mc_forward_impl(mc_input_BK)
        mc_output_B_K = BayesianModule.unflatten_tensor(
            mc_output_BK, k
        )  # B*kx... --> Bxkx...
        return mc_output_B_K

    def mc_forward_impl(self, mc_input_BK: torch.Tensor) -> torch.Tensor:
        return mc_input_BK

    def det_forward_impl(self, input: torch.Tensor) -> torch.Tensor:
        return input

    @staticmethod
    def unflatten_tensor(x: torch.Tensor, k: int):
        """Transform Tensor of shape B*kx --> Bxk"""
        x = x.view([-1, BayesianModule.k, *x.shape[1:]])
        return x

    @staticmethod
    def flatten_tensor(x: torch.Tensor):
        """Transform Tensor of shape Bxk --> B*kx"""
        return x.flatten(0, 1)

    @staticmethod
    def mc_tensor(x: torch.Tensor, k: int):
        """Copy every sample k times Bx... --> B*kx..."""
        mc_shape = [x.shape[0], k, *x.shape[1:]]
        return x.unsqueeze(1).expand(mc_shape).flatten(0, 1)


class ConsistentMCDropout(nn.Module):
    def __init__(self, p: float = 0.5):
        """MC Dropout module with internal state for masking operation.
        Subsequent iterations when model is in validation mode mask the same inputs.
        This is required for BatchBALD.

        Args:
            p (float, optional): Dropout p for Bernoulli distribution. Defaults to 0.5.
        """
        super().__init__()
        self.p = p
        self._mask = None

    def forward(self, x: torch.Tensor):
        if self.training:
            return F.dropout(x, p=self.p, training=True, inplace=False)
        else:
            if self._mask is None or self._mask.shape[1] != BayesianModule.k:
                self._build_mask(x, BayesianModule.k)
            x = x.view(
                [
                    -1,
                    BayesianModule.k,
                ]
                + list(x.shape[1:])
            )  # B x  k x ...
            x = x.masked_fill(self._mask, 0) / (1 - self.p)
            x = x.flatten(0, 1)  # NK
            return x

    def _build_mask(self, x, k):
        self._mask = torch.empty(
            [1, k] + list(x.shape[1:]), dtype=torch.bool, device=x.device
        ).bernoulli(p=self.p)

    def reset(self):
        self._mask = None

    def eval(self):
        self.reset()
        return super().eval()


class ConsistenMCDropout2D(ConsistentMCDropout):
    def _build_mask(self, x, k):
        self._mask = torch.empty(
            [1, k] + list(x.shape[1:3]) + [1], dtype=torch.bool, device=x.device
        ).bernoulli(p=self.p)
