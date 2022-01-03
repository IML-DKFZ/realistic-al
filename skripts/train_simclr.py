import os

import hydra
from omegaconf import DictConfig

from pl_bolts.models.self_supervised.simclr.simclr_module import *
from pytorch_lightning import loggers

import sys
import os
import torch
import torchvision

from pytorch_lightning.loggers import TensorBoardLogger
from pl_bolts.callbacks.ssl_online import SSLOnlineEvaluator
from pl_bolts.models.self_supervised.simclr.transforms import (
    SimCLREvalDataTransform,
    SimCLRTrainDataTransform,
)

src_folder = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"
)

sys.path.append(src_folder)
from data.data import TorchVisionDM
import utils
from utils.config_utils import print_config
from models.networks import build_wideresnet


"""
For Cifar10 Training:
python train_simclr.py ++trainer.max_epochs 1000
"""


class SimCLR_algo(SimCLR):
    def init_model(self):
        if self.arch in ["resnet18", "resnet50"]:
            return super().init_model()

        #  Implementation of WideResnet is ugly but will do for now
        # TODO - Possible: Change this with additional changes to models!
        elif "wideresnet" in self.arch:
            params = self.arch.split("wideresnet")[1]
            depth, widen_factor = params.split("-")
            wideresnet = build_wideresnet(
                int(depth), int(widen_factor), dropout=0, num_classes=1
            )
            wideresnet.linear = torch.nn.Identity()

            wideresnet.forward = lambda x: [wideresnet.get_features(x)]
            return wideresnet
        else:
            raise NotImplementedError


@hydra.main(config_path="./config", config_name="simclr_base")
def cli_cluster(cfg: DictConfig):
    print_config(cfg, fields=["model", "trainer", "data"])
    utils.set_seed(cfg.trainer.seed)

    dm = TorchVisionDM(
        data_root=cfg.base.data_root,
        batch_size=cfg.model.batch_size,
        dataset=cfg.data.dataset,
        val_split=cfg.data.val_split,
        random_split=True,
        active=False,
        num_classes=cfg.data.num_classes,
        mean=cfg.data.mean,
        std=cfg.data.std,
        # transforms are overwritten later
        transform_train=cfg.data.transform_train,
        transform_test=cfg.data.transform_test,
        shape=cfg.data.shape,
    )

    normalization = torchvision.transforms.Normalize(cfg.data.mean, cfg.data.std)

    dm.train_set.transform = SimCLRTrainDataTransform(
        input_height=cfg.data.shape[0],
        gaussian_blur=cfg.model.gaussian_blur,
        jitter_strength=cfg.model.jitter_strength,
        normalize=normalization,
    )

    dm.val_set.transform = SimCLREvalDataTransform(
        input_height=cfg.data.shape[0],
        gaussian_blur=cfg.model.gaussian_blur,
        jitter_strength=cfg.model.jitter_strength,
        normalize=normalization,
    )

    run_dict = dict(**cfg.model, **cfg.trainer, **cfg.data)
    run_dict["num_samples"] = len(dm.train_set)

    model = SimCLR_algo(**run_dict)

    online_evaluator = None
    if cfg.model.online_ft:
        # online eval
        online_evaluator = SSLOnlineEvaluator(
            drop_p=0.0,
            hidden_dim=None,
            z_dim=cfg.model.hidden_mlp,
            num_classes=cfg.data.num_classes,
            dataset=cfg.data.dataset,
        )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_checkpoint = ModelCheckpoint(save_last=True, save_top_k=1, monitor="val_loss")
    callbacks = (
        [model_checkpoint, online_evaluator]
        if cfg.model.online_ft
        else [model_checkpoint]
    )
    callbacks.append(lr_monitor)

    tb_logger = TensorBoardLogger(
        save_dir=cfg.trainer.experiments_root,
        name=cfg.trainer.experiment_name,
        version=cfg.trainer.experiment_id,
    )

    trainer = Trainer(
        gpus=cfg.trainer.gpus,
        distributed_backend="ddp" if cfg.trainer.gpus > 1 else None,
        sync_batchnorm=True if cfg.trainer.gpus > 1 else False,
        logger=tb_logger,
        max_epochs=cfg.trainer.max_epochs,
        fast_dev_run=cfg.trainer.fast_dev_run,
        terminate_on_nan=True,
        callbacks=callbacks,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        progress_bar_refresh_rate=cfg.trainer.progress_bar_refresh_rate,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        precision=cfg.trainer.precision,
        benchmark=cfg.trainer.deterministic is False,
        deterministic=cfg.trainer.deterministic,
        # enable_progress_bar=cfg.trainer.enable_progress_bar,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    cli_cluster()
