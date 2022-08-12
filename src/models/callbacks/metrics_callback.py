import pytorch_lightning as pl
from pytorch_lightning import Callback
from torchmetrics import AUROC, AveragePrecision


class ISICMetricCallback(Callback):
    def __init__(self):
        """Callback which creates and tracks the torchmetrics AUROC and Average Prescision.
        """
        super().__init__()
        modes = ["train", "val", "test"]
        self.metric_dict = {f"{mode}/auroc": AUROC(num_classes=2) for mode in modes}
        self.metric_dict.update(
            {f"{mode}/av_prec": AveragePrecision(num_classes=2) for mode in modes}
        )

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, unused: int = 0
    ) -> None:
        mode = "train"
        logprob = outputs["logprob"]
        y = outputs["label"]
        # _, logprob, y = outputs
        self.update_metric_dict(pl_module, mode, logprob, y)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, unused: int = 0
    ) -> None:
        mode = "val"
        logprob, y = outputs
        self.update_metric_dict(pl_module, mode, logprob, y)

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, unused: int = 0
    ) -> None:
        mode = "test"
        logprob, y = outputs
        self.update_metric_dict(pl_module, mode, logprob, y)

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        mode = "train"
        self.reset_metric_dict(pl_module, mode)

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        mode = "val"
        self.reset_metric_dict(pl_module, mode)

    def on_test_epoch_start(self, trainer, pl_module) -> None:
        mode = "test"
        self.reset_metric_dict(pl_module, mode)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        mode = "train"
        self.log_metric_dict(pl_module, mode=mode)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        mode = "val"
        self.log_metric_dict(pl_module, mode=mode)

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        mode = "test"
        self.log_metric_dict(pl_module, mode=mode)

    def log_metric_dict(self, pl_module, mode):
        # metric_dict = pl_module.metric_dict
        metric_dict = self.metric_dict
        for key in metric_dict:
            if key.split("/")[0] == mode:
                pl_module.log(
                    key, metric_dict[key].compute(), on_step=False, on_epoch=True
                )

    def reset_metric_dict(self, pl_module, mode):
        # metric_dict = pl_module.metric_dict
        metric_dict = self.metric_dict
        for key in metric_dict:
            if key.split("/")[0] == mode:
                metric_dict[key].reset()

    def update_metric_dict(self, pl_module, mode, preds, y):
        # metric_dict = pl_module.metric_dict
        metric_dict = self.metric_dict
        for key in metric_dict:
            if key.split("/")[0] == mode:
                metric_dict[key].update(preds, y)
