import torch

import pytorch_lightning as pl
from pytorch_lightning import Callback
from torchmetrics import (
    AUROC,
    AveragePrecision,
    ConfusionMatrix,
)


class MetricCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module) -> None:
        mode = "train"
        self.reset_metric_dict(self.auc_dict, pl_module, mode)
        self.reset_metric_dict(self.pred_dict, pl_module, mode)
        self.reset_metric_dict(self.pred_conf, pl_module, mode)

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        mode = "val"
        self.reset_metric_dict(self.auc_dict, pl_module, mode)
        self.reset_metric_dict(self.pred_dict, pl_module, mode)
        self.reset_metric_dict(self.pred_conf, pl_module, mode)

    def on_test_epoch_start(self, trainer, pl_module) -> None:
        mode = "test"
        self.reset_metric_dict(self.auc_dict, pl_module, mode)
        self.reset_metric_dict(self.pred_dict, pl_module, mode)
        self.reset_metric_dict(self.pred_conf, pl_module, mode)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        mode = "train"
        self.log_metric_dict(self.auc_dict, pl_module, mode=mode)
        self.log_metric_dict(self.pred_dict, pl_module, mode=mode)
        self.log_metrics(pl_module, mode)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        mode = "val"
        self.log_metric_dict(self.auc_dict, pl_module, mode=mode)
        self.log_metric_dict(self.pred_dict, pl_module, mode=mode)
        self.log_metrics(pl_module, mode)

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        mode = "test"
        self.log_metric_dict(self.auc_dict, pl_module, mode=mode)
        self.log_metric_dict(self.pred_dict, pl_module, mode=mode)
        self.log_metrics(pl_module, mode)

    def log_metrics(self, pl_module, mode):
        pass

    @staticmethod
    def log_metric_dict(metric_dict, pl_module, mode):
        # metric_dict = pl_module.metric_dict
        # metric_dict = self.auc_dict
        for key in metric_dict:
            if key.split("/")[0] == mode:
                pl_module.log(
                    key, metric_dict[key].compute(), on_step=False, on_epoch=True
                )

    @staticmethod
    def reset_metric_dict(metric_dict, pl_module, mode):
        # metric_dict = pl_module.metric_dict
        # metric_dict = self.auc_dict
        for key in metric_dict:
            if key.split("/")[0] == mode:
                metric_dict[key].reset()

    @staticmethod
    def update_metric_dict(metric_dict, pl_module, mode, preds, y):
        # metric_dict = pl_module.metric_dict
        # metric_dict = self.auc_dict
        for key in metric_dict:
            if key.split("/")[0] == mode:
                metric_dict[key].update(preds, y)


class ImbClassMetricCallback(MetricCallback):
    def __init__(self, num_classes):
        """Callback which creates and tracks the torchmetrics AUROC and Average Prescision.
        Note: balanced multiclass accuracy = mean of diagonal of conf matrix, divided by positive incidences
        """
        super().__init__()
        modes = ["train", "val", "test"]
        self.pred_dict = {}
        self.pred_conf = {
            mode: ConfusionMatrix(num_classes=num_classes, normalize=None)
            for mode in modes
        }

        self.auc_dict = {}
        self.auc_dict.update(
            {
                f"{mode}/auroc": AUROC(num_classes=num_classes, mode="macro")
                for mode in modes
            }
        )
        self.auc_dict.update(
            {
                f"{mode}/av_prec": AveragePrecision(
                    num_classes=num_classes, mode="macro"
                )
                for mode in modes
            }
        )

    def compute_pred_metrics(self, mode):
        conf_mat = self.pred_conf[mode].compute()
        acc = conf_mat.diag().sum() / conf_mat.sum()
        w_acc = (conf_mat.diag() / conf_mat.sum(dim=1)).mean()
        av_prec = (conf_mat.diag() / conf_mat.sum(dim=0)).mean()
        av_f1 = (
            2 * conf_mat.diag() / (conf_mat.sum(dim=1) + conf_mat.sum(dim=0))
        ).mean()

        return {
            f"{mode}/w_acc": w_acc,
            f"{mode}/av_prec": av_prec,
            f"{mode}/av_f1": av_f1,
        }

    def log_metrics(self, pl_module, mode):
        if mode in self.pred_conf:
            log_dict = self.compute_pred_metrics(mode)
            pl_module.log_dict(log_dict, on_epoch=True, on_step=False)

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, unused: int = 0
    ) -> None:
        mode = "train"
        logprob = outputs["logprob"]
        y = outputs["label"]
        pred = torch.argmax(logprob, dim=-1)
        self.update_metric_dict(
            self.pred_conf, pl_module, mode, pred.to("cpu"), y.to("cpu")
        )
        self.update_metric_dict(self.auc_dict, pl_module, mode, torch.exp(logprob), y)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, unused: int = 0
    ) -> None:
        mode = "val"
        logprob, y = outputs
        pred = torch.argmax(logprob, dim=-1)
        self.update_metric_dict(
            self.pred_conf, pl_module, mode, pred.to("cpu"), y.to("cpu")
        )
        self.update_metric_dict(self.auc_dict, pl_module, mode, torch.exp(logprob), y)

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, unused: int = 0
    ) -> None:
        mode = "test"
        logprob, y = outputs
        pred = torch.argmax(logprob, dim=-1)
        self.update_metric_dict(
            self.pred_conf, pl_module, mode, pred.to("cpu"), y.to("cpu")
        )
        self.update_metric_dict(self.auc_dict, pl_module, mode, torch.exp(logprob), y)


class ISIC2016MetricCallback(MetricCallback):
    def __init__(self):
        """Callback which creates and tracks the torchmetrics AUROC and Average Prescision.
        """
        super().__init__()
        modes = ["train", "val", "test"]
        self.auc_dict = {f"{mode}/auroc": AUROC(num_classes=2) for mode in modes}
        self.auc_dict.update(
            {f"{mode}/av_prec": AveragePrecision(num_classes=2) for mode in modes}
        )
        self.pred_dict = {}
        self.pred_conf = {}

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, unused: int = 0
    ) -> None:
        mode = "train"
        logprob = outputs["logprob"]
        y = outputs["label"]
        self.update_metric_dict(self.auc_dict, pl_module, mode, torch.exp(logprob), y)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, unused: int = 0
    ) -> None:
        mode = "val"
        logprob, y = outputs
        self.update_metric_dict(self.auc_dict, pl_module, mode, torch.exp(logprob), y)

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, unused: int = 0
    ) -> None:
        mode = "test"
        logprob, y = outputs
        self.update_metric_dict(self.auc_dict, pl_module, mode, torch.exp(logprob), y)
