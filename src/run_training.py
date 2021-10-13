import pytorch_lightning as pl
from torch.functional import norm
from torch.utils import data
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
from typing import Union

from utils.storing import ActiveStore


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
    return data, labels


def train(cfg: DictConfig):
    cfg.trainer.seed = pl.utilities.seed.seed_everything(cfg.trainer.seed)

    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir=cfg.trainer.experiments_root,
        name=cfg.trainer.experiment_name,
        version=cfg.trainer.experiment_id,
    )

    datamodule = TorchVisionDM(
        data_root=cfg.trainer.data_root,
        batch_size=cfg.trainer.batch_size,
    )
    datamodule.prepare_data()
    datamodule.setup("fit")
    # datamodule.train_set.label_randomly(1000)
    datamodule.train_set.label_balanced(10, 10)

    model = BayesianModule(config=cfg)

    lr_monitor = pl.callbacks.LearningRateMonitor()

    trainer = pl.Trainer(
        gpus=cfg.trainer.n_gpus,
        logger=tb_logger,
        # max_steps=cfg.trainer.n_steps,
        max_epochs=cfg.trainer.max_epochs,
        fast_dev_run=cfg.trainer.fast_dev_run,
        terminate_on_nan=True,
        callbacks=[lr_monitor],
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        progress_bar_refresh_rate=cfg.trainer.progress_bar_refresh_rate,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
    )
    # train_loader = datamodule.train_dataloader()
    # val_loader = datamodule.val_dataloader()
    # test_loader = datamodule.test_dataloader()
    # print('-'*8)
    # from tqdm import tqdm
    # print(f'Lenth of the Training Loader: {len(train_loader)}') # TODO: why does the length of the dataloader change from here
    # print('Iterating over Train Loader:')
    # for x in tqdm(train_loader):
    #     pass
    # print('-'*8)
    # trainer.fit(model=model, datamodule=datamodule,train_dataloaders=train_loader, val_dataloaders=val_loader) # to the fit where it is used? --> look into that
    # trainer.test(test_dataloaders=test_loader)

    trainer.fit(model=model, datamodule=datamodule)
    test_results = trainer.test()
    return active_callback(model, datamodule)


def training_loop(
    cfg, count: Union[None, int] = None, datamodule: TorchVisionDM = None
):

    model = BayesianModule(config=cfg)

    if count is not None:
        version = cfg.trainer.experiment_id
    else:
        version = "{}_loop-{}".format(cfg.trainer.experiment_id, count)
    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir=cfg.trainer.experiments_root,
        name=cfg.trainer.experiment_name,
        version=version,
    )

    lr_monitor = pl.callbacks.LearningRateMonitor()

    trainer = pl.Trainer(
        gpus=cfg.trainer.n_gpus,
        logger=tb_logger,
        # max_steps=cfg.trainer.n_steps,
        max_epochs=cfg.trainer.max_epochs,
        fast_dev_run=cfg.trainer.fast_dev_run,
        terminate_on_nan=True,
        callbacks=[lr_monitor],
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        progress_bar_refresh_rate=cfg.trainer.progress_bar_refresh_rate,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
    )
    trainer.fit(model=model, datamodule=datamodule)
    test_results = trainer.test()
    return active_callback(model, datamodule)


def active_callback(
    model: pl.LightningModule,
    datamodule: TorchVisionDM,
    acq_size: int = 10,
):
    pool = datamodule.train_set.pool
    model = model.to("cuda:0")
    acq_function = get_bald_fct(model, k=200)
    pool_loader = datamodule.pool_dataloader(batch_size=64)
    acq_vals, acq_inds = query_sampler(pool_loader, acq_function, num_queries=acq_size)

    acq_data, acq_labels = obtain_data_from_pool(pool, acq_inds)

    vis_path = "/home/c817h/Documents/projects/Active_Learning/activeframework/visuals"
    fig, axs = plots.visualize_samples(plots.normalize(acq_data), acq_vals)
    plt.savefig(os.path.join(vis_path, "labeled_samples.pdf"))
    plt.clf()

    fig, axs = plots.visualize_labels(acq_labels)
    plt.savefig(os.path.join(vis_path, "labelled_targets.pdf"))
    print("-" * 8)
    print("Evaluating Validation Set")
    accuracy_val = evaluate_accuracy(model, datamodule.val_dataloader())
    print("-" * 8)
    print("Evaluating Training Set")
    accuracy_test = evaluate_accuracy(model, datamodule.test_dataloader())

    return ActiveStore(
        requests=acq_inds,
        n_labelled=datamodule.train_set.n_labelled,
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
        correct += (pred.cpu() == y).sum()
        counts += y.shape[0]
    return correct / counts


if __name__ == "__main__":
    main()
