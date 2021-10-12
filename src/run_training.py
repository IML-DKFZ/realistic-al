import pytorch_lightning as pl
from torch.functional import norm
from models.bayesian import Bayesian_Module
from data import TorchVisionDM
import hydra
from omegaconf import DictConfig
from utils import config_utils
from query_sampler import query_sampler, get_bald_fct, get_bay_entropy_fct
import torch
from utils import plots
import matplotlib.pyplot as plt
import os



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

    lr_monitor = pl.callbacks.LearningRateMonitor()

    datamodule = TorchVisionDM(
        data_root=cfg.trainer.data_root,
        batch_size=cfg.trainer.batch_size,
    )
    datamodule.prepare_data()
    datamodule.setup("fit")
    # datamodule.train_set.label_randomly(1000)
    datamodule.train_set.label_balanced(10, 10)

    model = Bayesian_Module(config=cfg)
    trainer = pl.Trainer(
        gpus=cfg.trainer.n_gpus,
        logger=tb_logger,
        # max_steps=cfg.trainer.n_steps,
        max_epochs = cfg.trainer.max_epochs,
        fast_dev_run=cfg.trainer.fast_dev_run,
        terminate_on_nan=True,
        callbacks=[lr_monitor],
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        progress_bar_refresh_rate=cfg.trainer.progress_bar_refresh_rate,
        gradient_clip_val=1,
    )
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()
    print('-'*8)
    print(len(train_loader)) # TODO: why does the length of the dataloader change from here
    print('-'*8)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader) # to the fit where it is used? --> look into that

    trainer.test(test_dataloaders=test_loader)
    pool_loader = datamodule.pool_dataloader(batch_size=64)
    pool = datamodule.train_set.pool
    model=model.to('cuda:0')
    acq_function = get_bald_fct(model, k=200)
    acq_vals, acq_inds = query_sampler(pool_loader, acq_function, num_queries=1000)

    acq_data, acq_labels = obtain_data_from_pool(pool, acq_inds)


    vis_path = '/home/c817h/Documents/projects/Active_Learning/activeframework/visuals'
    fig, axs = plots.visualize_samples(plots.normalize(acq_data), acq_vals)
    plt.savefig(os.path.join(vis_path,'labeled_samples.pdf'))
    plt.clf()

    fig, axs = plots.visualize_labels(acq_labels)
    plt.savefig(os.path.join(vis_path,'labelled_targets.pdf'))






if __name__ == "__main__":
    main()
