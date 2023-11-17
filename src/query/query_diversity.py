import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from torch.utils.data import DataLoader

from models.bayesian_module import BayesianModule

from .kcenterGreedy import KCenterGreedy

names = """kcentergreedy badge""".split()

DEVICE = "cuda:0"


def query_sampler(cfg, model, labeled_dataloader, unlabeled_dataloader, acq_size):
    name = cfg.query.name
    if name == "kcentergreedy":
        indices = get_kcg(
            model, labeled_dataloader, unlabeled_dataloader, acq_size=acq_size
        )
        return indices, np.arange(acq_size)[::-1]
    elif name == "badge":
        indices = get_badge(
            model, labeled_dataloader, unlabeled_dataloader, acq_size=acq_size
        )
        return indices, np.arange(acq_size)[::-1]
    else:
        raise NotImplementedError


def init_centers(X: np.ndarray, K: int, device: str):
    """
    Source: https://github.com/decile-team/distil/blob/main/distil/active_learning_strategies/badge.py

    Args:
        X (np.ndarray): _description_
        K (int): _description_
        device (_type_): _description_

    Returns:
        _type_: _description_
    """
    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    mu = [X[ind]]
    indsAll = [ind]
    centInds = [0.0] * len(X)
    cent = 0
    while len(mu) < K:
        if len(mu) == 1:
            D2 = (
                chunked_pdist(torch.from_numpy(X), torch.from_numpy(mu[-1]))
                .numpy()
                .astype(float)
            )
        else:
            newD = (
                chunked_pdist(torch.from_numpy(X), torch.from_numpy(mu[-1]))
                .numpy()
                .astype(float)
            )
            for i in range(len(X)):
                if D2[i] > newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]

        # if sum(D2) == 0.0: pdb.set_trace()
        D2 = D2.ravel().astype(float)
        Ddist = (D2**2) / sum(D2**2)
        customDist = stats.rv_discrete(name="custm", values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    return indsAll


def chunked_pdist(
    x: torch.Tensor, y: torch.Tensor, device: str = "cuda:0", max_size: int = 8
):
    pdist = torch.nn.PairwiseDistance(p=2)
    new_shape = x.shape[0]
    while (x.element_size() * x.nelement() / 10e9) * (
        new_shape / x.shape[0]
    ) > max_size:
        new_shape /= 2

    y_dev = y.to(device)
    ind_start = 0
    new_shape = x.shape[0] / 2
    dists = []
    while ind_start < x.shape[0]:
        ind_offset = int(min(new_shape, x.shape[0] - ind_start))
        dist_part = pdist(x[ind_start : ind_start + ind_offset].to("cuda"), y_dev).to(
            "cpu"
        )
        dists.append(dist_part)
        ind_start += ind_offset
    dists = torch.concat(dists)
    assert dists.shape[0] == x.shape[0]
    return dists


def get_grad_embedding(
    model, dataloader: DataLoader, grad_embedding_type="linear"
) -> torch.Tensor:
    BayesianModule.k = 1
    start_index = 0
    for i, (x, y) in enumerate(dataloader):
        inputs = x.to(DEVICE)
        with torch.no_grad():
            features = model.get_features(inputs)  # B x Z
        l1 = features  # B x Z
        if not model.hparams.model.small_head:
            l1 = model.model.classifier[:-1](features)
        embDim = l1.shape[-1]
        outputs = model.model.classifier(features)

        preds = torch.argmax(outputs, dim=1)

        loss = F.cross_entropy(outputs, preds, reduction="sum")
        l0_grads = torch.autograd.grad(loss, outputs)[0]  # B x C

        # Calculate the linear layer gradients as well if needed
        if grad_embedding_type != "bias":
            l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
            l1_grads = l0_expand * l1.repeat(1, outputs.shape[-1])  # B x C*Z

        # Populate embedding tensor according to the supplied argument.
        if grad_embedding_type == "bias":
            gradient_embedding = l0_grads
        elif grad_embedding_type == "linear":
            gradient_embedding = l1_grads
        else:
            gradient_embedding = torch.cat([l0_grads, l1_grads], dim=1)
        if i == 0:
            gradient_embeddings = torch.empty(
                [len(dataloader.dataset), gradient_embedding.shape[1]], device="cpu"
            )
        gradient_embeddings[
            start_index : start_index + gradient_embedding.shape[0]
        ] = gradient_embedding.to("cpu")
        start_index += gradient_embedding.shape[0]
        torch.cuda.empty_cache()
    return gradient_embeddings


def get_badge(model, labeled_dataloader, pool_loader, acq_size=100):
    grad_embedding = get_grad_embedding(model, pool_loader)
    grad_embedding = grad_embedding.numpy()
    acq_indices = init_centers(grad_embedding, acq_size, DEVICE)
    return np.array(acq_indices)


def get_kcg(model, labeled_dataloader, pool_loader, acq_size=100):
    """Returns the indices of the core-set for the pool of the model via the k-center Greedy approach."""
    with torch.no_grad():
        features = torch.tensor([]).to(DEVICE)
        for inputs, _ in labeled_dataloader:
            inputs = inputs.to(DEVICE)
            features_batch = model.get_features(inputs)
            features = torch.cat((features, features_batch), 0)
        feat_labeled = features.detach().cpu().numpy()

        features = torch.tensor([]).to(DEVICE)
        for inputs, _ in pool_loader:
            inputs = inputs.to(DEVICE)
            features_batch = model.get_features(inputs)
            features = torch.cat((features, features_batch), 0)
        feat_unlabeled = features.detach().cpu().numpy()

        feat_merge = np.concatenate([feat_labeled, feat_unlabeled], axis=0)
        indices_labeled = np.arange(feat_labeled.shape[0])
        del features
        del feat_labeled
        del feat_unlabeled
        sampling = KCenterGreedy(feat_merge)
        acq_indices = np.array(sampling.select_batch_(indices_labeled, acq_size))

        # subtract the indices of the labeled data to get pool indices
        acq_indices -= indices_labeled.shape[0]
    return acq_indices
