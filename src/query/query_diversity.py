import numpy as np
import torch

from .kcenterGreedy import KCenterGreedy

names = """kcentergreedy""".split()

DEVICE = "cuda:0"


def query_sampler(cfg, model, labeled_dataloader, unlabeled_dataloader, acq_size):
    name = cfg.query.name
    if name == "kcentergreedy":
        indices = get_kcg(
            model, labeled_dataloader, unlabeled_dataloader, acq_size=acq_size
        )
        return indices, np.arange(acq_size)[::-1]
    else:
        raise NotImplementedError


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


# from https://github.com/cubeyoung/TA-VAAL
#
# def get_kcg(models, labeled_data_size, unlabeled_loader):
#     models['backbone'].eval()
#     with torch.cuda.device(CUDA_VISIBLE_DEVICES):
#         features = torch.tensor([]).cuda()

#     with torch.no_grad():
#         for inputs, _, _ in unlabeled_loader:
#             with torch.cuda.device(CUDA_VISIBLE_DEVICES):
#                 inputs = inputs.cuda()
#             _, features_batch, _ = models['backbone'](inputs)
#             features = torch.cat((features, features_batch), 0)
#         feat = features.detach().cpu().numpy()
#         new_av_idx = np.arange(SUBSET,(SUBSET + labeled_data_size))
#         sampling = kCenterGreedy(feat)
#         batch = sampling.select_batch_(new_av_idx, ADDENDUM)
#         other_idx = [x for x in range(SUBSET) if x not in batch]
#     return  other_idx + batch
