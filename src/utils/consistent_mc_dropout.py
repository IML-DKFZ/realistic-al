from abc import abstractclassmethod
import torch 
import torch.nn as nn
import torch.nn.functional as F


class BayesianModule(nn.Module):
    k=None
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, k:int):
        """Standard Forward Pass of the module."""
        BayesianModule.k =k

        mc_input_BK = BayesianModule.mc_tensor(x, k) # Bx .... --> B*kx ...
        mc_output_BK = self.mc_forward_impl(mc_input_BK) 
        mc_output_B_K = BayesianModule.unflatten_tensor(mc_output_BK, k) #B*kx... --> Bxkx...
        return mc_output_B_K

    def mc_forward_impl(self, mc_input_BK: torch.Tensor):
        return mc_input_BK


    @staticmethod
    def unflatten_tensor(x: torch.Tensor, k:int):
        """Transform Tensor of shape B*kx --> Bxk"""
        x = x.view([-1,  BayesianModule.k, *x.shape[1:]])
        return x 
    
    @staticmethod
    def flatten_tensor(x: torch.Tensor):
        """Transform Tensor of shape Bxk --> B*kx"""
        return x.flatten(0,1)
    
    @staticmethod
    def mc_tensor(x: torch.Tensor, k: int):
        """Copy every sample k times Bx... --> B*kx..."""
        mc_shape = [x.shape[0], k, *x.shape[1:]]
        return x.unsqueeze(1).expand(mc_shape).flatten(0, 1)



class ConsistentMCDropout(nn.Module):
    def __init__(self, p:float = 0.5):
        super().__init__()
        self.p = p
        self._mask = None

    def forward(self, x:torch.Tensor):
        if self.training:
            return F.dropout(x, p=self.p, training=True, inplace=False)
        else:
            if self._mask is None:
                self._build_mask(x,  BayesianModule.k)
            x = x.view([-1,  BayesianModule.k,] + list(x.shape[1:])) # B x  k x ...
            x = x.masked_fill(self._mask, 0)/(1-self.p)
            x = x.flatten(0, 1) # NK
            return x

    def _build_mask(self, x, k):
        self._mask = torch.empty([1, k] + list(x.shape[1:]), dtype=torch.bool, device=x.device).bernoulli(p=self.p)

    def reset(self):
        self._mask = None 

    def eval(self):
        self.reset()
        return super().eval()



class ConsistenMCDropout2D(ConsistentMCDropout):
    def _build_mask(self, x, k):
        self._mask = torch.empty([1, k] + list(x.shape[1:3]) + [1], dtype=torch.bool, device=x.device).bernoulli(p=self.p)