import pytorch_lightning as pl
from torch.functional import norm
from models.bayesian import BayesianModule
from data import TorchVisionDM
import hydra
from omegaconf import DictConfig
from utils import config_utils
from query_sampler import query_sampler, get_bald_fct, get_bay_entropy_fct
import torch
from utils import plots
import matplotlib.pyplot as plt
import os
from run_training import training_loop
import math


@hydra.main(config_path="./config", config_name="config")
def main(cfg: DictConfig):
    config_utils.print_config(cfg)

    active_loop(
        cfg,
        cfg.active.num_labelled,
        cfg.active.balanced,
        cfg.active.acq_size,
        cfg.active.num_iter,
    )


def active_loop(
    cfg: DictConfig,
    num_labelled: int = 100,
    balanced: bool = True,
    acq_size: int = 10,
    num_iter: int = 0,
):
    datamodule = TorchVisionDM(
        data_root=cfg.trainer.data_root,
        batch_size=cfg.trainer.batch_size,
        dataset=cfg.data.name,
    )
    datamodule.prepare_data()
    datamodule.setup()
    num_classes = 10
    if balanced:
        datamodule.train_set.label_balanced(
            n_per_class=num_labelled // num_classes, num_classes=num_classes
        )
    else:
        datamodule.train_set.label_randomly(num_labelled)

    if num_iter == 0:
        num_iter = math.ceil(len(datamodule.train_set) / acq_size)

    active_stores = []
    for i in range(num_iter):
        active_store = training_loop(cfg, count=i, datamodule=datamodule)
        # active_store = train(cfg, datamodule)
        datamodule.train_set.label(active_store.requests)
        active_stores.append(active_store)

    import matplotlib.pyplot as plt

    accs = [active_store.accuracy_val for active_store in active_stores]
    num_samples = [active_store.n_labelled for active_store in active_stores]
    vis_path = "/home/c817h/Documents/projects/Active_Learning/activeframework/visuals"
    plt.plot(num_samples, accs)
    plt.savefig(os.path.join(vis_path, "accs_vs_num_samples.pdf"))
    # add later evaluation


if __name__ == "__main__":
    main()
