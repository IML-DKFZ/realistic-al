import math
import numpy as np
import torch

import torch.nn.functional as F

DEVICE = "cuda:0"
###


def get_acq_function(cfg, pt_model):
    name = str(cfg.active.name).split("_")[0]
    if name == "bald":
        return get_bald_fct(pt_model, cfg.active.k)
    elif name == "entropy":
        return get_bay_entropy_fct(pt_model, cfg.active.k)
    elif name == "random":
        return get_random_fct()
    elif name == "batchbald":
        return get_bay_logits(pt_model, cfg.active.k)
    else:
        raise NotImplementedError


def get_post_acq_function(cfg):
    names = str(cfg.active.name).split("_")[0]
    if cfg.active.name == "batchbald":
        from batchbald_redux.batchbald import get_batchbald_batch

        # This values should only be used to select the entropy computation
        # TODO: verify this!
        # num_samples = 100000 # taken from batchbald_redux notebook --> old bb
        num_samples = 40000  # taken from BatchBALD

        def post_acq_function(logprob_n_k_c: np.ndarray, num_queries: int):
            """BatchBALD acquisition function using logits with iterative conditional mutual information."""
            assert (
                len(logprob_n_k_c.shape) == 3
            )  # make sure that input is of correct type
            logprob_n_k_c = torch.from_numpy(logprob_n_k_c).to(
                device=DEVICE, dtype=torch.double
            )
            out = get_batchbald_batch(
                logprob_n_k_c,
                batch_size=num_queries,
                num_samples=num_samples,
                dtype=torch.double,
                device=DEVICE,
            )
            indices = np.array(out.indices)
            scores = np.array(out.scores)
            return indices, scores

        return post_acq_function
    elif len(names) == 2 and names[-1] == "random":

        def post_acq_function(
            acq_scores: np.ndarray,
            num_queries: int,
            subset_size: int = cfg.active.subset,
        ):
            """Acquires a random subset of scores and then select based on ranking."""
            assert len(acq_scores.shape) == 1
            subset_size = min(subset_size, len(acq_scores))  # size of the subset
            acq_ind = np.arange(len(acq_scores))
            inds = np.argsort(acq_scores)[::-1]
            inds = np.random.choice(inds, size=subset_size, replace=False)
            inds = inds[:num_queries]
            acq_list = acq_scores[inds]
            acq_ind = acq_ind[inds]
            return inds, acq_list

    else:

        def post_acq_function(acq_scores: np.ndarray, num_queries: int):
            """Acquires based on ranking."""
            assert len(acq_scores.shape) == 1  # make sure that input is of correct type
            acq_ind = np.arange(len(acq_scores))
            inds = np.argsort(acq_scores)[::-1]
            inds = inds[:num_queries]
            acq_list = acq_scores[inds]
            acq_ind = acq_ind[inds]
            return inds, acq_list

        return post_acq_function


###


def query_sampler(dataloader, acq_function, post_acq_function, num_queries=64):
    """Returns the queries (acquistion values and indices) given the data pool and the acquisition function.
    The Acquisition Function Returns Numpy arrays!"""
    acq_list = []
    for i, batch in enumerate(dataloader):
        acq_values = acq_from_batch(batch, acq_function, device=DEVICE)
        acq_list.append(acq_values)
    acq_list = np.concatenate(acq_list)
    acq_ind, acq_scores = post_acq_function(acq_list, num_queries)

    return acq_scores, acq_ind


def sample_acq_fct(batch):
    x, y = batch
    scores = np.arange(x.shape[0])
    return scores


def get_bay_entropy_fct(pt_model, k=5):
    def acq_bay_entropy(x: torch.Tensor):
        """Returns the Entropy of predictions of the bayesian model"""
        with torch.no_grad():
            out = pt_model(x, k=k, agg=False)  # BxkxD
            ent = bay_entropy(out)
        return ent

    return acq_bay_entropy


def get_bald_fct(pt_model, k=5):
    def acq_bald(x: torch.Tensor):
        """Returns the BALD-acq values (Mutual Information) between most likely labels and the model parameters"""
        with torch.no_grad():
            out = pt_model(x, k=k, agg=False)
            mut_info = mutual_bald(out)
        return mut_info

    return acq_bald


def get_bay_logits(pt_model, k=5):
    def acq_logits(x: torch.Tensor):
        """Returns the NxKxC logprobs needed for BatchBALD"""
        with torch.no_grad():
            out = pt_model(x, k=k, agg=False)
            out = torch.log_softmax(out, dim=2)
        return out

    return acq_logits


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
    k = logits.shape[1]
    out = F.log_softmax(logits, dim=2)  # BxkxD
    # This part was wrong but it performed better than BatchBALD - interesting
    # out = out.mean(dim=1)  # BxD
    out = torch.logsumexp(out, dim=1) - math.log(k)  # BxkxD --> BxD
    ent = torch.sum(-torch.exp(out) * out, dim=1)  # B
    return ent


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
