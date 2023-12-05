import os
import sys

import hydra
import torch
import torchvision
from omegaconf import DictConfig
from pl_bolts.callbacks.ssl_online import SSLOnlineEvaluator
from pl_bolts.models.self_supervised.evaluator import SSLEvaluator
from pl_bolts.models.self_supervised.simclr.simclr_module import *
from pl_bolts.models.self_supervised.simclr.transforms import (
    SimCLREvalDataTransform,
    SimCLRTrainDataTransform,
)
from pytorch_lightning import loggers
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_warn

src_folder = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"
)

sys.path.append(src_folder)
import utils
from data.data import TorchVisionDM
from models.networks import build_wideresnet
from utils.config_utils import print_config

"""
For Cifar10 Training:
python train_simclr.py ++trainer.max_epochs 1000
"""


class SyncFunction_DDP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]

        gathered_tensor = [
            torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())
        ]

        torch.distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone().contiguous()
        torch.distributed.all_reduce(
            grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False
        )

        idx_from = torch.distributed.get_rank() * ctx.batch_size
        idx_to = (torch.distributed.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to]


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

    def nt_xent_loss(self, out_1, out_2, temperature, eps=1e-6):
        """
        assume out_1 and out_2 are normalized
        out_1: [batch_size, dim]
        out_2: [batch_size, dim]
        """
        # gather representations in case of distributed training
        # out_1_dist: [batch_size * world_size, dim]
        # out_2_dist: [batch_size * world_size, dim]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            out_1_dist = SyncFunction_DDP.apply(out_1)
            out_2_dist = SyncFunction_DDP.apply(out_2)
        else:
            out_1_dist = out_1
            out_2_dist = out_2

        # out: [2 * batch_size, dim]
        # out_dist: [2 * batch_size * world_size, dim]
        out = torch.cat([out_1, out_2], dim=0)
        out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)

        # cov and sim: [2 * batch_size, 2 * batch_size * world_size]
        # neg: [2 * batch_size]
        cov = torch.mm(out.contiguous(), out_dist.t().contiguous())
        sim = torch.exp(cov / temperature)
        neg = sim.sum(dim=-1)

        # from each row, subtract e^(1/temp) to remove similarity measure for x1.x1
        row_sub = Tensor(neg.shape).fill_(math.e ** (1 / temperature)).to(neg.device)
        neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

        # Positive similarity, pos becomes [2 * batch_size]
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)

        loss = -torch.log(pos / (neg + eps)).mean()

        return loss


class SSLOnlineEvaluatorDDP(SSLOnlineEvaluator):
    def on_pretrain_routine_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        # must move to device after setup, as during setup, pl_module is still on cpu
        self.online_evaluator = SSLEvaluator(
            n_input=self.z_dim,
            n_classes=self.num_classes,
            p=self.drop_p,
            n_hidden=self.hidden_dim,
        ).to(pl_module.device)

        # switch fo PL compatibility reasons
        accel = (
            trainer.accelerator_connector
            if hasattr(trainer, "accelerator_connector")
            else trainer._accelerator_connector
        )
        if accel.is_distributed:
            # if accel.use_ddp:
            from torch.nn.parallel import DistributedDataParallel as DDP

            self.online_evaluator = DDP(
                self.online_evaluator, device_ids=[pl_module.device]
            )
        # elif accel.use_dp:
        #     from torch.nn.parallel import DataParallel as DP

        #     self.online_evaluator = DP(
        #         self.online_evaluator, device_ids=[pl_module.device]
        #     )
        # else:
        #     rank_zero_warn(
        #         "Does not support this type of distributed accelerator. The online evaluator will not sync."
        #     )

        self.optimizer = torch.optim.Adam(self.online_evaluator.parameters(), lr=1e-4)

        if self._recovered_callback_state is not None:
            self.online_evaluator.load_state_dict(
                self._recovered_callback_state["state_dict"]
            )
            self.optimizer.load_state_dict(
                self._recovered_callback_state["optimizer_state"]
            )


@hydra.main(config_path="./config", config_name="simclr_base", version_base="1.1")
def cli_cluster(cfg: DictConfig):
    print_config(cfg, fields=["model", "trainer", "data"])
    utils.set_seed(cfg.trainer.seed)

    imbalance = False
    if "imbalance" in cfg.data:
        imbalance = cfg.data.imbalance

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
        seed=cfg.trainer.seed,
        imbalance=imbalance,
        persistent_workers=True,
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
        online_evaluator = SSLOnlineEvaluatorDDP(
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
        accelerator="gpu",
        devices=cfg.trainer.gpus,
        sync_batchnorm=True if cfg.trainer.gpus > 1 else False,
        logger=tb_logger,
        max_epochs=cfg.trainer.max_epochs,
        fast_dev_run=cfg.trainer.fast_dev_run,
        strategy="ddp" if cfg.trainer.gpus not in [0, 1] else None,
        detect_anomaly=True,
        callbacks=callbacks,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        progress_bar_refresh_rate=cfg.trainer.progress_bar_refresh_rate,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        precision=cfg.trainer.precision,
        benchmark=cfg.trainer.deterministic is False,
        deterministic=cfg.trainer.deterministic,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    cli_cluster()
