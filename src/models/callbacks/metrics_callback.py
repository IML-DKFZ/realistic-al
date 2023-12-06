from typing import Any, Dict

import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback
from torchmetrics import AUROC, AveragePrecision, ConfusionMatrix


class MetricCallback(Callback):
    """Class for using using multiple different metrics including automated resets.
    IMPLEMENTATION: add additional metrics here."""

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        mode = "train"
        self.reset_metric_dict(self.auc_dict, mode)
        self.reset_metric_dict(self.pred_dict, mode)
        self.reset_metric_dict(self.pred_conf, mode)

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        mode = "val"
        self.reset_metric_dict(self.auc_dict, mode)
        self.reset_metric_dict(self.pred_dict, mode)
        self.reset_metric_dict(self.pred_conf, mode)

    def on_test_epoch_start(self, trainer, pl_module) -> None:
        mode = "test"
        self.reset_metric_dict(self.auc_dict, mode)
        self.reset_metric_dict(self.pred_dict, mode)
        self.reset_metric_dict(self.pred_conf, mode)

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

    def log_metrics(self, pl_module: pl.LightningModule, mode: str) -> None:
        pass

    @staticmethod
    def log_metric_dict(
        metric_dict: Dict[str, Any], pl_module: pl.LightningModule, mode: str
    ) -> None:
        """Log values in metric dict with correct mode to pytorch lightning module logger.

        Args:
            metric_dict (Dict[str, Any]): dictionary with keys: {mode/metric_name: metric}
            pl_module (pl.LightningModule): module used
            mode (str): prefix name of logdict e.g. train, val, test
        """
        for key in metric_dict:
            if key.split("/")[0] == mode:
                pl_module.log(
                    key, metric_dict[key].compute(), on_step=False, on_epoch=True
                )

    @staticmethod
    def reset_metric_dict(metric_dict: Dict[str, Any], mode: str) -> None:
        """Reset metrics in metric dict for a specific mode.

        Args:
            metric_dict (Dict[str, Any]): dictionary with keys: {mode/metric_name: metric}
            mode (str): prefix name of logdict e.g. train, val, test
        """
        for key in metric_dict:
            if key.split("/")[0] == mode:
                metric_dict[key].reset()

    @staticmethod
    def update_metric_dict(
        metric_dict: Dict[str, Any], mode: str, preds: torch.Tensor, y: torch.Tensor
    ):
        """Update metrics in metric dict for a specific mode.

        Args:
            metric_dict (Dict[str, Any]): dictionary with keys: {mode/metric_name: metric}
            mode (str): prefix name of logdict e.g. train, val, test
            preds (torch.Tensor): predictions
            y (torch.Tensor): labels
        """
        for key in metric_dict:
            if key.split("/")[0] == mode:
                metric_dict[key].update(preds, y)


### IMPLEMENTATION
# New callbacks for metrics can be implemented here following the examples of ImbClassMetricCallback


class ImbClassMetricCallback(MetricCallback):
    def __init__(self, num_classes: int):
        """Callback which creates and tracks the torchmetrics AUROC and Average Prescision and values from Confusion Matrix (balanced accuracy, precision, F1 etc.).

        Args:
            num_classes (int): #of classes in dataset
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

    def compute_pred_metrics(
        self, mode: str, class_wise: bool = False
    ) -> Dict[str, float]:
        """Compute prediction values based on confusion matrix.
        When class_wise is true, return rec, prec and f1 for every class.

        Args:
            mode (str): prefix name of logdict e.g. train, val, test
            class_wise (bool, optional): add values per class. Defaults to False.

        Returns:
            dict: values for logging in subsequent steps
        """
        conf_mat = self.pred_conf[mode].compute()
        acc = conf_mat.diag().sum() / conf_mat.sum()
        # Note: balanced multiclass accuracy = mean of diagonal of conf matrix, divided by positive incidences
        w_acc = (conf_mat.diag() / conf_mat.sum(dim=1)).mean()
        av_prec = (conf_mat.diag() / conf_mat.sum(dim=0)).mean()
        av_f1 = (
            2 * conf_mat.diag() / (conf_mat.sum(dim=1) + conf_mat.sum(dim=0))
        ).mean()

        out_dict = {
            f"{mode}/w_acc": w_acc,
            f"{mode}/av_prec": av_prec,
            f"{mode}/av_f1": av_f1,
        }
        if class_wise:
            class_wise_dict = {
                "rec": conf_mat.diag() / conf_mat.sum(dim=1),
                "prec": conf_mat.diag() / conf_mat.sum(dim=0),
                "f1": 2 * conf_mat.diag() / (conf_mat.sum(dim=1) + conf_mat.sum(dim=0)),
            }
            for metric in class_wise_dict:
                for cls, val in enumerate(class_wise_dict[metric]):
                    out_dict[f"{mode}/{metric}/cls_{cls}"] = class_wise_dict[metric][
                        cls
                    ]

        return out_dict

    def log_metrics(self, pl_module: pl.LightningModule, mode: str):
        """Compute log_metrics and write them to loggers.

        Args:
            pl_module (pl.LightningModule): Module for training
            mode (str): prefix name of logdict e.g. train, val, test
        """
        if mode in self.pred_conf:
            log_dict = self.compute_pred_metrics(mode)
            pl_module.log_dict(log_dict, on_epoch=True, on_step=False)

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs: Dict[str, torch.Tensor],
        batch,
        batch_idx,
        unused: int = 0,
    ) -> None:
        """Update confusion matrix and auc dicts with data from train batch.

        Args:
            outputs (Dict[str, torch.Tensor]): contains keys [logprob, label]
            ...
        """
        mode = "train"
        logprob = outputs["logprob"]
        y = outputs["label"]
        pred = torch.argmax(logprob, dim=-1)
        self.update_metric_dict(self.pred_conf, mode, pred.to("cpu"), y.to("cpu"))
        self.update_metric_dict(self.auc_dict, mode, torch.exp(logprob), y)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, unused: int = 0
    ) -> None:
        """Update confusion matrix and auc dicts with data from validation batch.

        Args:
            outputs (Dict[str, torch.Tensor]): contains keys [logprob, label]
            ...
        """
        mode = "val"
        logprob, y = outputs
        pred = torch.argmax(logprob, dim=-1)
        self.update_metric_dict(self.pred_conf, mode, pred.to("cpu"), y.to("cpu"))
        self.update_metric_dict(self.auc_dict, mode, torch.exp(logprob), y)

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, unused: int = 0
    ) -> None:
        """Update confusion matrix and auc dicts with data from test batch.

        Args:
            outputs (Dict[str, torch.Tensor]): contains keys [logprob, label]
            ...
        """
        mode = "test"
        logprob, y = outputs
        pred = torch.argmax(logprob, dim=-1)
        self.update_metric_dict(self.pred_conf, mode, pred.to("cpu"), y.to("cpu"))
        self.update_metric_dict(self.auc_dict, mode, torch.exp(logprob), y)


# This code is not used in benchmark, but maybe it is useful for some people.
class ISIC2016MetricCallback(MetricCallback):
    def __init__(self):
        """Callback which creates and tracks the torchmetrics AUROC and Average Prescision."""
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
        self.update_metric_dict(self.auc_dict, mode, torch.exp(logprob), y)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, unused: int = 0
    ) -> None:
        mode = "val"
        logprob, y = outputs
        self.update_metric_dict(self.auc_dict, mode, torch.exp(logprob), y)

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, unused: int = 0
    ) -> None:
        mode = "test"
        logprob, y = outputs
        self.update_metric_dict(self.auc_dict, mode, torch.exp(logprob), y)
