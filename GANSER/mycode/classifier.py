import logging
from typing import Any, Dict, List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection

_EVALUATE_OUTPUT = List[Dict[str, float]]  # 1 dict per DataLoader

log = logging.getLogger('torcheeg')


def classification_metrics(metric_list: List[str], num_classes: int):
    allowed_metrics = [
        'precision', 'recall', 'f1score', 'accuracy', 'matthews', 'auroc',
        'kappa'
    ]

    for metric in metric_list:
        if metric not in allowed_metrics:
            raise ValueError(
                f"{metric} is not allowed. Please choose 'precision', 'recall', 'f1score', 'accuracy', 'matthews', 'auroc', 'kappa'."
            )
    metric_dict = {
        'accuracy':
        torchmetrics.Accuracy(task='multiclass',
                              num_classes=num_classes,
                              top_k=1),
        'precision':
        torchmetrics.Precision(task='multiclass',
                               average='macro',
                               num_classes=num_classes),
        'recall':
        torchmetrics.Recall(task='multiclass',
                            average='macro',
                            num_classes=num_classes),
        'f1score':
        torchmetrics.F1Score(task='multiclass',
                             average='macro',
                             num_classes=num_classes),
        'matthews':
        torchmetrics.MatthewsCorrCoef(task='multiclass',
                                      num_classes=num_classes),
        'auroc':
        torchmetrics.AUROC(task='multiclass', num_classes=num_classes),
        'kappa':
        torchmetrics.CohenKappa(task='multiclass', num_classes=num_classes)
    }
    metrics = [metric_dict[name] for name in metric_list]
    return MetricCollection(metrics)


class ClassifierTrainer(pl.LightningModule):
    r'''
        A generic trainer class for EEG classification.
        The model is expected to return a tuple: (outputs, features), where
        outputs: the logits used for computing loss and metrics,
        features: the intermediate features extracted from the network.
    '''

    def __init__(self,
                 model: nn.Module,
                 num_classes: int,
                 lr: float = 1e-3,
                 weight_decay: float = 0.0,
                 devices: int = 1,
                 accelerator: str = "cpu",
                 metrics: List[str] = ["accuracy"]):

        super().__init__()
        self.model = model

        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay

        self.devices = devices
        self.accelerator = accelerator
        self.metrics = metrics

        self.ce_fn = nn.CrossEntropyLoss()

        self.init_metrics(metrics, num_classes)

    def init_metrics(self, metrics: List[str], num_classes: int) -> None:
        self.train_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()
        self.test_loss = torchmetrics.MeanMetric()

        self.train_metrics = classification_metrics(metrics, num_classes)
        self.val_metrics = classification_metrics(metrics, num_classes)
        self.test_metrics = classification_metrics(metrics, num_classes)

    def fit(self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            max_epochs: int = 300,
            *args,
            **kwargs) -> Any:
        trainer = pl.Trainer(devices=self.devices,
                             accelerator=self.accelerator,
                             max_epochs=max_epochs,
                             enable_progress_bar=True,  # 关闭进度条
                             logger=False,               # 关闭内置日志记录
                             enable_checkpointing=False,
                             *args,
                             **kwargs)
        return trainer.fit(self, train_loader, val_loader)

    def test(self, test_loader: DataLoader, *args,
             **kwargs) -> _EVALUATE_OUTPUT:
        trainer = pl.Trainer(devices=self.devices,
                             enable_checkpointing=False,
                             accelerator=self.accelerator,
                             *args,
                             **kwargs)
        return trainer.test(self, test_loader)

    # 修改 forward，要求模型返回 (outputs, features)
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor],
                      batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)  # 解包返回的 tuple
        loss = self.ce_fn(y_hat, y)

        self.log("train_loss",
                 self.train_loss(loss),
                 prog_bar=True,
                 on_epoch=False,
                 logger=False,
                 on_step=True)

        for i, metric_value in enumerate(self.train_metrics.values()):
            self.log(f"train_{self.metrics[i]}",
                     metric_value(y_hat, y),
                     prog_bar=True,
                     on_epoch=False,
                     logger=False,
                     on_step=True)

        # 这里可根据需要使用 features 进行额外处理
        return loss

    def on_train_epoch_end(self) -> None:
        self.log("train_loss",
                 self.train_loss.compute(),
                 prog_bar=False,
                 on_epoch=True,
                 on_step=False,
                 logger=True)
        for i, metric_value in enumerate(self.train_metrics.values()):
            self.log(f"train_{self.metrics[i]}",
                     metric_value.compute(),
                     prog_bar=False,
                     on_epoch=True,
                     on_step=False,
                     logger=True)

        # print the metrics
        msg = "\n[Train] "
        for key, value in self.trainer.logged_metrics.items():
            if key.startswith("train_"):
                msg += f"{key}: {value:.3f} "
        log.info(msg + '\n')

        self.train_loss.reset()
        self.train_metrics.reset()

    def validation_step(self, batch: Tuple[torch.Tensor],
                        batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)  # 修改为解包
        loss = self.ce_fn(y_hat, y)

        self.val_loss.update(loss)
        self.val_metrics.update(y_hat, y)
        return loss

    def on_validation_epoch_end(self) -> None:
        self.log("val_loss",
                 self.val_loss.compute(),
                 prog_bar=False,
                 on_epoch=True,
                 on_step=False,
                 logger=True)
        for i, metric_value in enumerate(self.val_metrics.values()):
            self.log(f"val_{self.metrics[i]}",
                     metric_value.compute(),
                     prog_bar=False,
                     on_epoch=True,
                     on_step=False,
                     logger=True)

        msg = "\n[Val] "
        for key, value in self.trainer.logged_metrics.items():
            if key.startswith("val_"):
                msg += f"{key}: {value:.3f} "
        log.info(msg + '\n')

        self.val_loss.reset()
        self.val_metrics.reset()

    def test_step(self, batch: Tuple[torch.Tensor],
                  batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat= self(x)  # 修改为解包
        loss = self.ce_fn(y_hat, y)

        self.test_loss.update(loss)
        self.test_metrics.update(y_hat, y)
        return loss

    def on_test_epoch_end(self) -> None:
        self.log("test_loss",
                 self.test_loss.compute(),
                 prog_bar=False,
                 on_epoch=True,
                 on_step=False,
                 logger=True)
        for i, metric_value in enumerate(self.test_metrics.values()):
            self.log(f"test_{self.metrics[i]}",
                     metric_value.compute(),
                     prog_bar=False,
                     on_epoch=True,
                     on_step=False,
                     logger=True)

        msg = "\n[Test] "
        for key, value in self.trainer.logged_metrics.items():
            if key.startswith("test_"):
                msg += f"{key}: {value:.3f} "
        log.info(msg + '\n')

        self.test_loss.reset()
        self.test_metrics.reset()

    def configure_optimizers(self):
        parameters = list(self.model.parameters())
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        optimizer = torch.optim.Adam(trainable_parameters,
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)
        return optimizer

    def predict_step(self,
                     batch: Tuple[torch.Tensor],
                     batch_idx: int,
                     dataloader_idx: int = 0):
        x, y = batch
        y_hat= self(x)  # 修改为解包
        return (y_hat, features)

    def save(self, param_path):
        torch.save({
            'c_model': self.model.state_dict(),
        }, param_path)
