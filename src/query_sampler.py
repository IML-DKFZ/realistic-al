import numpy as np
import torch

import torch.nn.functional as F

DEVICE = "cuda:0"


def query_sampler(dataloader, acq_function, num_queries=64):
    """Returns the queries (acquistion values and indices) given the data pool and the acquisition function.
    The Acquisition Function Returns Numpy arrays!"""
    acq_list = []
    for i, batch in enumerate(dataloader):
        acq_values = acq_from_batch(batch, acq_function, device=DEVICE)
        acq_list.append(acq_values)
    acq_list = np.concatenate(acq_list)
    acq_ind = np.arange(len(acq_list))
    inds = np.argsort(acq_list)[::-1]
    inds = inds[:num_queries]

    acq_list = acq_list[inds]
    acq_ind = acq_ind[inds]
    return acq_list, acq_ind


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
        """ "Returns the BALD-acq values (Mutual Information) between most likely labels and the model parameters"""
        with torch.no_grad():
            out = pt_model(x, k=k)
            mut_info = mutual_bald(out)
        return mut_info

    return acq_bald


def acq_from_batch(batch, function, device="cuda:0"):
    x, y = batch
    x = x.to(device)
    out = function(x)
    out = out.to("cpu").numpy()
    return out
