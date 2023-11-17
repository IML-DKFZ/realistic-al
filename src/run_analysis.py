from copy import deepcopy
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
import torch
from loguru import logger
from omegaconf import DictConfig
from sklearn.metrics import pairwise_distances
from torch.utils.data import DataLoader

import utils
from data.active import ActiveSubset
from models.bayesian import BayesianModule
from models.fixmatch import FixMatch
from query import QuerySampler
from query.kcenterGreedy import KCenterGreedy
from query.query_uncertainty import (get_bald_fct, get_bay_entropy_fct,
                                     get_exp_entropy_fct)
from run_test import get_active_torchvision_dm
from utils.dict_utils import dict_2_df
from utils.io import load_omega_conf
from utils.tensor import to_numpy

DEVICE = "cuda:0"


def normalize(x, x_max=1, x_min=0):
    return (x - x_min) / (x_max - x_min)


def compute_minimal_cluster_distance(
    points: np.ndarray, clusters: np.ndarray, metric="euclidean"
):
    return pairwise_distances(points, clusters, metric=metric).min(axis=1)


############# Current State ##############
# 1. Put AnalyzeExperiment into loop X
# 2. check whether for 1 experiments the results are identical X
# 3. self.train_batch_dicts is not nice implemented! X working
# 3.1 Test on MioTCD
# 4. Clean Up


QUERY_LIST = ["bald", "kcentergreedy", "entropy", "random"]


class AnalyzeExperiment:
    def __init__(self, path, dm=None, short: bool = False):
        self.path = path
        self.short = short
        self.cfg = load_omega_conf(path / "hparams.yaml")
        utils.set_seed(self.cfg.trainer.seed)
        if dm is None:
            dm = get_active_torchvision_dm(self.cfg)
        try:
            dm_cur = deepcopy(dm)
            dm_cur.train_set.load_checkpoint(path / "data_ckpt.pkl")
            dm = deepcopy(dm)
        except:
            pass
        self.dm = dm
        del dm_cur

        self.model = BayesianModule(self.cfg)
        self.model.eval()
        self.model.to(DEVICE)
        ckpt_path = self.model.get_best_ckpt(self.path)
        self.model.load_only_state_dict(ckpt_path)
        self.query_fct = self.cfg.query.name

        self.function_dict = {
            "MI": get_bald_fct(self.model),
            "Entropy": get_bay_entropy_fct(self.model),
            "Exp-Entropy": get_exp_entropy_fct(self.model),
            "Pred": lambda x: self.model.forward(x).argmax(dim=1),
            "Repr": lambda x: self.model.get_features(x),
        }
        self.batch_dicts = dict()
        self.sample_dicts = dict()

    @cached_property
    def train_sample_dict(self) -> Dict[str, np.ndarray]:
        train_sample_dict = obtain_function_values(
            self.dm.train_set.labelled_set, self.function_dict
        )
        distances = pairwise_distances(
            self.pool_sample_dict["Repr"], train_sample_dict["Repr"]
        ).min(axis=1)
        train_sample_dict["Cluster Distance"] = distances
        return train_sample_dict

    @cached_property
    def train_batch_dict(self) -> Dict[str, np.ndarray]:
        train_batch_dict = self.sample_to_batch_dict(self.train_sample_dict)
        distances = compute_minimal_cluster_distance(
            self.pool_sample_dict["Repr"], self.train_sample_dict["Repr"]
        )
        train_batch_dict["Maximum Cluster Distance"] = distances.max()
        return train_batch_dict

    @cached_property
    def pool_sample_dict(self) -> Dict[str, np.ndarray]:
        return obtain_function_values(self.dm.train_set.pool, self.function_dict)

    @cached_property
    def pool_batch_dict(self) -> Dict[str, np.ndarray]:
        return self.sample_to_batch_dict(self.pool_sample_dict)

    def compute_query_vals(self, query_name):
        logger.info("Computing Values for Query: {}".format(query_name))
        if query_name == "kcentergreedy":
            feat_merge = np.concatenate(
                [self.train_sample_dict["Repr"], self.pool_sample_dict["Repr"]], axis=0
            )
            indices_labeled = np.arange(self.train_sample_dict["Repr"].shape[0])
            sampling = KCenterGreedy(feat_merge)
            inds = np.array(
                sampling.select_batch_(indices_labeled, self.cfg.active.acq_size)
            )
            # subtract the indices of the labeled data to get pool indices
            inds -= indices_labeled.shape[0]
        elif query_name == "bald":
            inds = self.pool_sample_dict["MI"].argsort()[::-1]
            inds = inds[: self.cfg.active.acq_size]
        elif query_name == "entropy":
            inds = self.pool_sample_dict["Entropy"].argsort()[::-1]
            inds = inds[: self.cfg.active.acq_size]
        elif query_name == "random":
            inds = np.random.choice(
                np.arange(len(self.pool_sample_dict["Entropy"])),
                size=self.cfg.active.acq_size,
                replace=False,
            )
        else:
            logger.info(
                "QuerySampler Computing Values for Query: {}".format(query_name)
            )
            cfg = deepcopy(self.cfg)
            cfg.query.name = query_name
            query_sampler = QuerySampler(cfg, model=self.model, device=DEVICE)
            query_sampler.setup()
            stored = query_sampler.active_callback(self.dm)
            inds = stored.requests
        query_sample_dict = dict()
        for key in self.pool_sample_dict:
            query_sample_dict[key] = self.pool_sample_dict[key][inds]
        rep = query_sample_dict["Repr"]
        rep_dis = pairwise_distances(rep, rep)
        rep_dis_mean = np.sum(rep_dis, axis=1) / (rep_dis.shape[0] - 1)
        for i in range(len(rep_dis)):
            rep_dis[i, i] = np.NaN
        rep_dis_min = np.nanmin(rep_dis, axis=1)

        query_sample_dict["Batch Distance"] = rep_dis_mean
        query_sample_dict["Minimum Batch Distance"] = rep_dis_min

        query_sample_dict["Cluster Distance"] = self.compute_minimal_cluster_distance(
            inds
        )
        query_batch_dict = self.sample_to_batch_dict(query_sample_dict)
        query_batch_dict["Maximum Cluster Distance"] = query_sample_dict[
            "Cluster Distance"
        ].max()
        return query_batch_dict, query_sample_dict, inds

    def compute_minimal_cluster_distance(self, inds_label: np.array = None):
        if inds_label is not None:
            unqueried_ind = np.ones(len(self.pool_sample_dict["Repr"]), dtype=bool)
            unqueried_ind[inds_label] = 0
            distances = compute_minimal_cluster_distance(
                self.pool_sample_dict["Repr"][
                    unqueried_ind
                ],  # data not labeled from pool
                np.concatenate(
                    [
                        self.pool_sample_dict["Repr"][
                            ~unqueried_ind
                        ],  # pool labeled from acq
                        self.train_sample_dict["Repr"],
                    ]
                ),
            )
        else:
            raise NotImplementedError
        return distances

    def compute_final_dict(self):
        logger.info("Running Analysis on Experiment:\n{}".format(self.path))
        train_sample_dict = self.train_sample_dict
        pool_sample_dict = self.pool_sample_dict
        acq_batch_dict, acq_sample_dict, acq_ind = self.compute_query_vals(
            self.cfg.query.name
        )

        self.batch_dicts[self.query_fct] = acq_batch_dict
        cfg = deepcopy(self.cfg)
        if not self.short:
            for query_fct in QUERY_LIST:
                if query_fct == self.query_fct:
                    continue
                cfg.query.name = query_fct
                batch_dict, sample_dict, _ = self.compute_query_vals(cfg.query.name)
                self.batch_dicts[query_fct] = batch_dict

            self.add_scaled(key="Cluster Distance", max_scale_key="kcentergreedy")
            self.add_scaled(
                key="Maximum Cluster Distance", max_scale_key="kcentergreedy"
            )
            self.add_scaled(key="MI", max_scale_key="bald")
            self.add_scaled(key="Entropy", max_scale_key="entropy")
        self.batch_dicts["Train"] = self.train_batch_dict
        self.batch_dicts["Pool"] = self.pool_batch_dict
        return acq_ind

    def add_scaled(self, key: str, max_scale_key: str):
        for query_fct in self.batch_dicts:
            self.batch_dicts[query_fct][key + " Scaled"] = normalize(
                self.batch_dicts[query_fct][key],
                self.batch_dicts[max_scale_key][key],
                self.train_batch_dict[key],
            )

    def sample_to_batch_dict(self, sample_dict):
        batch_dict = dict()
        for key in sample_dict:
            if key in ["Label", "Repr", "Pred"]:
                continue
            batch_dict[key] = sample_dict[key].mean()
        return batch_dict


def main(cfg: DictConfig, path: Path):
    utils.set_seed(cfg.trainer.seed)
    dm = get_active_torchvision_dm(cfg)
    dm.train_set.load_checkpoint(path / "data_ckpt.pkl")
    model = BayesianModule(cfg)
    # TODO: Add loading of FixMatch model
    model.load_only_state_dict(path / "checkpoints" / "epoch=15-step=1376.ckpt")

    model = model.to(DEVICE)
    return compute_query_values(cfg, dm, model)


def main_active(path: Path, all_acq: bool = True):
    iterations = [sub_path for sub_path in path.iterdir() if "loop" in sub_path.name]
    iterations.sort(key=lambda x: x.name.split("-")[1])

    cfgs = [load_omega_conf(iteration / "hparams.yaml") for iteration in iterations]

    cfg = cfgs[0]
    utils.set_seed(cfg.trainer.seed)
    dm = get_active_torchvision_dm(cfg)
    dm_cur = deepcopy(dm)
    save_dict = dict()

    for i, (iteration, cfg) in enumerate(zip(iterations, cfgs)):
        analyze = AnalyzeExperiment(iteration, dm_cur, short=False)
        acq_indices = analyze.compute_final_dict()
        batch_dicts = analyze.batch_dicts
        dm_cur.train_set.label(acq_indices)
        save_dict[len(analyze.train_sample_dict["Pred"])] = batch_dicts
    save_dict = {iteration.parent.name: save_dict}
    dataframe_dict = dict_2_df(
        save_dict, col_names=["Experiment", "Num_Labels", "Acquisition"]
    )
    df = pd.DataFrame(dataframe_dict)
    if all_acq:
        save_path = path / "analysis_comp.csv"
    else:
        save_path = path / "analysis-short.csv"

    print("Saving to {}".format(save_path))
    df.to_csv(save_path)


def compute_query_values(
    cfg, dm, model, function_dict, train_sample_dict, pool_sample_dict
):
    query_sampler = QuerySampler(cfg, model=model, device=DEVICE)
    query_sampler.setup()
    stored = query_sampler.active_callback(dm)

    QueriedDataset = ActiveSubset(dm.train_set.pool, stored.requests)

    query_sample_dict = obtain_function_values(QueriedDataset, function_dict)

    rep = query_sample_dict["Repr"]
    rep_dis = pairwise_distances(rep, rep)
    rep_dis = np.sum(rep_dis, axis=1) / (rep_dis.shape[0] - 1)

    query_sample_dict["Batch Distance"] = rep_dis

    query_batch_dict = dict()
    for key in query_sample_dict:
        if key in ["Label", "Repr", "Pred"]:
            continue
        query_batch_dict[key] = query_sample_dict[key].mean()

    unqueried_ind = np.ones(len(pool_sample_dict["Repr"]), dtype=bool)
    unqueried_ind[stored.requests] = 0

    distances = pairwise_distances(
        to_numpy(pool_sample_dict["Repr"][unqueried_ind]),
        np.concatenate(
            [to_numpy(query_sample_dict["Repr"]), train_sample_dict["Repr"]]
        ),
    )
    # get shortest distance for each sample to query
    distances = np.min(distances, axis=1)
    query_sample_dict["Cluster Distance"] = distances
    query_batch_dict["Cluster Distance"] = distances.mean()
    query_batch_dict["Maximum Cluster Distance"] = distances.max()

    return query_batch_dict, query_sample_dict, stored


def obtain_function_values(QueriedDataset: ActiveSubset, function_dict):
    QueriedLoader = DataLoader(
        QueriedDataset,
        batch_size=512,
        shuffle=False,
        num_workers=10,
        persistent_workers=False,
    )
    sample_dict = {}
    count = 0
    for i, (x, y) in enumerate(QueriedLoader):
        x = x.to(DEVICE)
        with torch.no_grad():
            for key, function in function_dict.items():
                out: torch.Tensor = function(x)
                if key not in sample_dict:
                    shape = out.shape
                    new_shape = (len(QueriedDataset), *shape[1:])
                    sample_dict[key] = np.zeros(new_shape)
                num_samples = len(out)
                sample_dict[key][count : count + num_samples] = to_numpy(out)[:]
        count += num_samples
    del QueriedLoader

    sample_dict["Label"] = QueriedDataset.targets

    sample_dict["Failure Label"] = sample_dict["Label"] != sample_dict["Pred"]

    return sample_dict


if __name__ == "__main__":
    import os

    from utils.io import load_omega_conf

    # path = Path(
    #     "/home/c817h/Documents/logs_cluster/activelearning/cifar100/active-cifar100_low/basic-pretrained_model-resnet_drop-0.5_aug-cifar_randaugment_acq-bald_ep-80_freeze-False_smallhead-False/2022-09-04_18-05-51-269506/loop-0"
    # )
    # cfg = load_omega_conf(path / "hparams.yaml")
    # cfg.trainer.data_root = os.getenv("DATA_ROOT")
    # cfg.model.load_pretrained = None
    # main(cfg, path)
    # path = Path(
    #     "/home/c817h/Documents/logs_cluster/activelearning/cifar100/active-cifar100_low/basic-pretrained_model-resnet_drop-0.5_aug-cifar_randaugment_acq-bald_ep-80_freeze-False_smallhead-False/2022-09-04_18-05-51-269506"
    # )
    # base_path = Path(
    #     "/home/c817h/Documents/logs_cluster/activelearning/cifar100/active-cifar100_low/basic-pretrained_model-resnet_drop-0.5_aug-cifar_randaugment_acq-bald_ep-80_freeze-False_smallhead-False"
    # )
    # Mio-TCD Paths
    # path = Path(
    #     "/home/c817h/network/Cluster-Experiments/activelearning/miotcd/active-miotcd_low/basic-pretrained_model-resnet_drop-0.5_aug-imagent_randaug_acq-bald_ep-80_freeze-False_smallhead-False/2022-09-13_23-52-00-902011"
    # ) # Done
    # path = Path(
    #     "/home/c817h/network/Cluster-Experiments/activelearning/miotcd/active-miotcd_low/basic-pretrained_model-resnet_drop-0.5_aug-imagent_randaug_acq-bald_ep-80_freeze-False_smallhead-False/2022-09-13_23-52-03-934658"
    # )  # Done
    # path = Path(
    #     "/home/c817h/network/Cluster-Experiments/activelearning/miotcd/active-miotcd_low/basic-pretrained_model-resnet_drop-0.5_aug-imagent_randaug_acq-bald_ep-80_freeze-False_smallhead-False/2022-09-13_23-52-08-120366"
    # )  # Running
    # # ISIC-2019
    # path = Path(
    #     "/home/c817h/network/Cluster-Experiments/activelearning/isic2019/active-isic19_high/basic-pretrained_model-resnet_drop-0.5_aug-isic_randaugment_acq-bald_ep-80_freeze-False_smallhead-False/2022-09-07_04-24-45-344474"
    # )
    base_path = Path(
        "/home/c817h/network/Cluster-Experiments/activelearning/isic2019/active-isic19_high/basic-pretrained_model-resnet_drop-0.5_aug-isic_randaugment_acq-bald_ep-80_freeze-False_smallhead-False"
    )
    for path in base_path.iterdir():
        # if True:
        if path.is_dir():
            main_active(path)
