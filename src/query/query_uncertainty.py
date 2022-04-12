import math
import numpy as np
import torch

import torch.nn.functional as F

from typing import Callable, Tuple
from .batchbald_redux.batchbald import get_batchbald_batch

# DEVICE = "cuda:0"
###

names = """bald entropy random batchbald variationratios""".split()


def get_acq_function(cfg, pt_model) -> Callable[[torch.Tensor], torch.Tensor]:
    name = str(cfg.query.name).split("_")[0]
    if name == "bald":
        return get_bald_fct(pt_model)
    elif name == "entropy":
        return get_bay_entropy_fct(pt_model)
    elif name == "random":
        return get_random_fct()
    elif name == "batchbald":
        return get_bay_logits(pt_model)
    elif name == "variationratios":
        return get_var_ratios(pt_model)
    else:
        raise NotImplementedError


def get_post_acq_function(
    cfg, device="cuda:0"
) -> Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    names = str(cfg.query.name).split("_")
    if cfg.query.name == "batchbald":

        # This values should only be used to select the entropy computation
        # TODO: verify this! -- true
        # num_samples = 100000 # taken from batchbald_redux notebook --> old bb
        num_samples = 40000  # taken from BatchBALD

        def post_acq_function(logprob_n_k_c: np.ndarray, acq_size: int):
            """BatchBALD acquisition function using logits with iterative conditional mutual information."""
            assert (
                len(logprob_n_k_c.shape) == 3
            )  # make sure that input is of correct type
            logprob_n_k_c = torch.from_numpy(logprob_n_k_c).to(
                device=device, dtype=torch.double
            )
            out = get_batchbald_batch(
                logprob_n_k_c,
                batch_size=acq_size,
                num_samples=num_samples,
                dtype=torch.double,
                device=device,
            )
            indices = np.array(out.indices)
            scores = np.array(out.scores)
            return indices, scores

        return post_acq_function
    else:

        def post_acq_function(acq_scores: np.ndarray, acq_size: int):
            """Acquires based on ranking. Highest ranks are acquired first."""
            assert len(acq_scores.shape) == 1  # make sure that input is of correct type
            acq_ind = np.arange(len(acq_scores))
            inds = np.argsort(acq_scores)[::-1]
            inds = inds[:acq_size]
            acq_list = acq_scores[inds]
            acq_ind = acq_ind[inds]
            return inds, acq_list

        return post_acq_function


###


def query_sampler(
    dataloader, acq_function, post_acq_function, acq_size=64, device="cuda:0"
):
    """Returns the queries (acquistion values and indices) given the data pool and the acquisition function.
    The Acquisition Function Returns Numpy arrays!"""
    acq_list = []
    for i, batch in enumerate(dataloader):
        acq_values = acq_from_batch(batch, acq_function, device=device)
        acq_list.append(acq_values)
    acq_list = np.concatenate(acq_list)
    acq_ind, acq_scores = post_acq_function(acq_list, acq_size)

    return acq_scores, acq_ind


# def sample_acq_fct(batch):
#     x, y = batch
#     scores = np.arange(x.shape[0])
#     return scores


def get_bay_entropy_fct(pt_model):
    def acq_bay_entropy(x: torch.Tensor):
        """Returns the Entropy of predictions of the bayesian model"""
        with torch.no_grad():
            out = pt_model(x, agg=False)  # BxkxD
            ent = bay_entropy(out)
        return ent

    return acq_bay_entropy


def get_bald_fct(pt_model):
    def acq_bald(x: torch.Tensor):
        """Returns the BALD-acq values (Mutual Information) between most likely labels and the model parameters"""
        with torch.no_grad():
            out = pt_model(x, agg=False)
            mut_info = mutual_bald(out)
        return mut_info

    return acq_bald


def get_bay_logits(pt_model):
    def acq_logits(x: torch.Tensor):
        """Returns the NxKxC logprobs needed for BatchBALD"""
        with torch.no_grad():
            out = pt_model(x, agg=False)
            out = torch.log_softmax(out, dim=2)
        return out

    return acq_logits


def get_var_ratios(pt_model):
    def acq_var_ratios(x: torch.Tensor):
        """Returns the variation ratio values."""
        with torch.no_grad():
            out = pt_model(x, agg=False)
            out = var_ratios(out)
        return out

    return acq_var_ratios


def get_random_fct():
    def acq_random(x: torch.Tensor, c: float = 0.0001):
        """Returns random values over the interval [0, c)"""
        out = torch.rand(x.shape[0], device=x.device) * c
        return out

    return acq_random


def get_model_features(pt_model):
    def get_features(x: torch.Tensor):
        with torch.no_grad:
            return pt_model.get_features(x)

    return get_features


def bay_entropy(logits):
    """Get the mean entropy of multiple logits."""
    k = logits.shape[1]
    out = F.log_softmax(logits, dim=2)  # BxkxD
    # This part was wrong but it performed better than BatchBALD - interesting
    # out = out.mean(dim=1)  # BxD
    out = torch.logsumexp(out, dim=1) - math.log(k)  # BxkxD --> BxD
    ent = torch.sum(-torch.exp(out) * out, dim=1)  # B
    return ent


def var_ratios(logits):
    k = logits.shape[1]
    out = F.log_softmax(logits, dim=2)  # BxkxD
    out = torch.logsumexp(out, dim=1) - math.log(k)  # BxkxD --> BxD
    out = 1 - torch.exp(out.max(dim=-1).values)  # B
    return out


def mean_entropy(logits):
    out = F.log_softmax(logits, dim=2)  # BxkxD
    out = torch.sum(-torch.exp(out) * out, dim=2)  # Bxk
    out = torch.mean(out, dim=1)
    return out


def mutual_bald(logits):
    return bay_entropy(logits) - mean_entropy(logits)


def acq_from_batch(batch, function, device="cuda:0"):
    x, y = batch
    x = x.to(device)
    out = function(x)
    out = out.to("cpu").numpy()
    return out
