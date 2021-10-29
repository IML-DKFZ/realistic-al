import pytorch_lightning as pl
from torch.functional import norm
from torch.utils import data
from models.bayesian import BayesianModule
from data import TorchVisionDM
import hydra
from omegaconf import DictConfig
from utils import config_utils
from query_sampler import query_sampler, get_acq_function, get_post_acq_function
import torch
from utils import plots
import matplotlib.pyplot as plt
import os
from typing import Union
import numpy as np
import gc

from typing import Callable, Tuple

import utils
from utils.storing import ActiveStore

num_samples = 100
num_classes = 10
balanced = True
active = False


@hydra.main(config_path="./config", config_name="config")
def main(cfg: DictConfig):
    config_utils.print_config(cfg)
    train(cfg)


def obtain_data_from_pool(pool, indices):
    data, labels = [], []
    for ind in indices:
        sample = pool[ind]
        data.append(sample[0])
        labels.append(sample[1])
    data = torch.stack(data, dim=0)
    labels = torch.tensor(labels, dtype=torch.int)
    labels = labels.numpy()
    data = data.numpy()
    return data, labels


def train(cfg: DictConfig):
    # cfg.trainer.seed = pl.utilities.seed.seed_everything(cfg.trainer.seed)
    utils.set_seed(cfg.trainer.seed)

    datamodule = TorchVisionDM(
        data_root=cfg.trainer.data_root,
        batch_size=cfg.trainer.batch_size,
        active=active,
    )

    datamodule.prepare_data()
    datamodule.setup("fit")
    # num_samples = datamodule.train_set.n_unlabelled

    # datamodule.train_set.label_randomly(num_samples)
    if active:
        if not balanced:
            datamodule.train_set.label_randomly(num_samples)
        else:
            datamodule.train_set.label_balanced(num_samples // num_classes, num_classes)

    active_store = training_loop(cfg, datamodule, active=active)


def training_loop(
    cfg: DictConfig,
    datamodule: TorchVisionDM,
    count: Union[None, int] = None,
    active: bool = True,
):
    utils.set_seed(cfg.trainer.seed)

    model = BayesianModule(config=cfg)

    if count is None:
        version = cfg.trainer.experiment_id
        name = cfg.trainer.experiment_name
    else:
        version = "loop-{}".format(count)
        name = "{}/{}".format(cfg.trainer.experiment_name, cfg.trainer.experiment_id)
    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir=cfg.trainer.experiments_root,
        name=name,
        version=version,
    )

    lr_monitor = pl.callbacks.LearningRateMonitor()
    callbacks = [lr_monitor]
    if datamodule.val_dataloader() is not None:
        # ckpt_callback = pl.callbacks.ModelCheckpoint(monitor="val/loss", mode="min")
        ckpt_callback = pl.callbacks.ModelCheckpoint(monitor="val/acc", mode="max")
    else:
        ckpt_callback = pl.callbacks.ModelCheckpoint(monitor="train/acc", mode="max")
    callbacks.append(ckpt_callback)
    if cfg.trainer.early_stop:
        early_stop_callback = pl.callbacks.EarlyStopping('val/acc', mode='max')
        callbacks.append(early_stop_callback)


    trainer = pl.Trainer(
        gpus=cfg.trainer.n_gpus,
        logger=tb_logger,
        max_epochs=cfg.trainer.max_epochs,
        fast_dev_run=cfg.trainer.fast_dev_run,
        terminate_on_nan=True,
        callbacks=callbacks,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        progress_bar_refresh_rate=cfg.trainer.progress_bar_refresh_rate,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
    )
    trainer.fit(model=model, datamodule=datamodule)
    best_path = ckpt_callback.best_model_path
    model.load_from_checkpoint(best_path)

    model = model.to("cuda:0")
    # TODO: Results from test and active callback for accuracy are NOT equal
    test_results = trainer.test(model=model)
    gc.collect()
    torch.cuda.empty_cache()

    model = model.to("cuda:0")
    model.eval()
    acq_function = get_acq_function(cfg, model)
    post_acq_function = get_post_acq_function(cfg)
    stored =  active_callback(
        model,
        datamodule,
        acq_function,
        post_acq_function=post_acq_function,
        count=count,
        acq_size=cfg.active.acq_size,
        active=active,
    )
    return stored


def active_callback(
    model: pl.LightningModule,
    datamodule: TorchVisionDM,
    acq_function: Callable[[torch.Tensor], torch.Tensor],
    post_acq_function: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
    acq_size: int = 10,
    count: Union[None, int] = None,
    active: bool = True,
):

    if active:
        pool = datamodule.train_set.pool
        pool_loader = datamodule.pool_dataloader(batch_size=64)
        acq_vals, acq_inds = query_sampler(
            pool_loader, acq_function,post_acq_function=post_acq_function, num_queries=acq_size
        )
        acq_data, acq_labels = obtain_data_from_pool(pool, acq_inds)
        n_labelled = datamodule.train_set.n_labelled

        suffix = ""
        if count is not None:
            suffix = f"_{count}"
        vis_path = "."
        fig, axs = plots.visualize_samples(plots.normalize(acq_data), acq_vals)
        plt.savefig(os.path.join(vis_path, "labeled_samples{}.pdf".format(suffix)))
        plt.clf()

        fig, axs = plots.visualize_labels(acq_labels, num_classes)
        plt.savefig(os.path.join(vis_path, "labelled_targets{}.pdf".format(suffix)))
        plt.clf()
    else:
        acq_inds = np.zeros(acq_size)
        acq_labels = np.zeros(acq_size)
        n_labelled = len(datamodule.train_set)

    print("-" * 8)
    print("Evaluating Validation Set")
    accuracy_val = evaluate_accuracy(model, datamodule.val_dataloader())
    print("-" * 8)
    print("Evaluating Training Set")
    accuracy_test = evaluate_accuracy(model, datamodule.test_dataloader())

    return ActiveStore(
        requests=acq_inds,
        n_labelled=n_labelled,
        accuracy_val=accuracy_val,
        accuracy_test=accuracy_test,
        labels=acq_labels,
    )


def evaluate_accuracy(model, dataloader):
    if dataloader is None:
        return 0
    counts = 0
    correct = 0
    for batch in dataloader:
        x, y = batch
        x = x.to("cuda:0")
        out = model(x)
        pred = torch.argmax(out, dim=1)
        correct += (pred.cpu() == y).sum().item()
        counts += y.shape[0]
    return correct / counts


if __name__ == "__main__":
    main()
