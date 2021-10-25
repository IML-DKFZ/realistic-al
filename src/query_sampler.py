import numpy as np
import torch

import torch.nn.functional as F

DEVICE = "cuda:0"
###


def get_acq_function(cfg, pt_model):
    if cfg.active.name == "bald":
        return get_bald_fct(pt_model, cfg.active.k)
    elif cfg.active.name == "entropy":
        return get_bay_entropy_fct(pt_model, cfg.active.k)
    elif cfg.active.name == "random":
        return get_random_fct()
    elif cfg.active.name == "batchbald":
        return get_bay_logits(pt_model, cfg.active.k)
    else:
        raise NotImplementedError


def get_post_acq_function(cfg):
    if cfg.active.name == "batchbald":
        from batchbald_redux.batchbald import get_batchbald_batch 
        # This values should only be used to select the entropy computation
        # TODO: verify this!
        num_samples = 100000 # taken from batchbald_redux notebook 
        def post_acq_function(logprob_n_k_c:np.ndarray, num_queries:int):
            assert len(logprob_n_k_c.shape) == 3 # make sure that input is of correct type
            logprob_n_k_c = torch.from_numpy(logprob_n_k_c).to(device=DEVICE, dtype=torch.double)
            out = get_batchbald_batch(logprob_n_k_c, batch_size=num_queries, num_samples=num_samples, dtype=torch.double, device=DEVICE)
            indices = np.ndarray(out.indices)
            scores = np.ndarray(out.scores)
            return indices, scores
        return post_acq_function
    else:
        def post_acq_function(acq_scores:np.ndarray, num_queries:int):
            assert len(acq_scores.shape) ==1 # make sure that input is of correct type 
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

    # acq_ind = np.arange(len(acq_list))
    # inds = np.argsort(acq_list)[::-1]
    # inds = inds[:num_queries]
    # acq_list = acq_list[inds]
    # acq_ind = acq_ind[inds]
    
    return acq_scores, acq_ind


def sample_acq_fct(batch):
    x, y = batch
    scores = np.arange(x.shape[0])
    return scores


def get_bay_entropy_fct(pt_model, k=5):
    def acq_bay_entropy(x: torch.Tensor):
        """Returns the Entropy of predictions of the bayesian model"""
        with torch.no_grad():
            out = pt_model(x, k=k)  # BxkxD
            ent = bay_entropy(out)
        return ent

    return acq_bay_entropy


def get_random_fct():
    def acq_random(x: torch.Tensor, c: float = 0.0001):
        """Returns random values over the interval [0, c)"""
        out = torch.rand(x.shape[0], device=x.device) * c
        return out

    return acq_random


def bay_entropy(logits):
    out = F.log_softmax(logits, dim=2)  # BxkxD
    out = out.mean(dim=1)  # BxD
    ent = torch.sum(-torch.exp(out) * out, dim=1)  # B
    return ent


def mean_entropy(logits):
    out = F.log_softmax(logits, dim=2)  # BxkxD
    out = torch.sum(-torch.exp(out) * out, dim=2)  # Bxk
    out = torch.mean(out, dim=1)
    return out


def mutual_bald(logits):
    return bay_entropy(logits) - mean_entropy(logits)


def get_bald_fct(pt_model, k=5):
    def acq_bald(x: torch.Tensor):
        """Returns the BALD-acq values (Mutual Information) between most likely labels and the model parameters"""
        with torch.no_grad():
            out = pt_model(x, k=k)
            mut_info = mutual_bald(out)
        return mut_info

    return acq_bald

def get_bay_logits(pt_model, k=5):
    def acq_logits(x:torch.Tensor):
        """Returns the NxKxC logits needed for BatchBALD"""
        with torch.no_grad():
            out = pt_model(x, k=k)
        return out
    return acq_logits


def acq_from_batch(batch, function, device="cuda:0"):
    x, y = batch
    x = x.to(device)
    out = function(x)
    out = out.to("cpu").numpy()
    return out
