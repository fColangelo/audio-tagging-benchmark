from typing import Any, List

import torch
from pytorch_lightning import LightningModule

from torchmetrics import MetricCollection
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification.confusion_matrix import ConfusionMatrix
from torchmetrics.classification.avg_precision import AveragePrecision
from torchmetrics.classification.auroc import AUROC
from utils.metrics import Lwlrap

import matplotlib.pyplot as plt
import seaborn as sns

from omegaconf.dictconfig import DictConfig
from hydra.utils import instantiate

class AudioTaggerModule(LightningModule):

    def __init__(
        self,
        cfg: DictConfig,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.cfg = cfg

        self.net = instantiate(self.cfg.model)

        # loss function
        self.criterion = instantiate(self.cfg.loss)
        
        if self.cfg.datamodule.dataset == "FSD50K":
            self.metrics = MetricCollection([
                AveragePrecision(num_classes=cfg.datamodule.num_classes),
                AUROC(num_classes=cfg.datamodule.num_classes),
                Lwlrap(num_classes=cfg.datamodule.num_classes),  
                ConfusionMatrix(num_classes=cfg.datamodule.num_classes, multilabel=True)
                ])
            # hardcoded for the 200 classes of FSD50K
            self.cm_figsize = 60
        else:
            self.metrics = MetricCollection([
            Accuracy(),
            ConfusionMatrix(num_classes=cfg.datamodule.num_classes)
            ])
            # hardcoded for the 50 classes of ESC and 10 of UrbanSound
            self.cm_figsize = 20 if self.cfg.datamodule.dataset=="ESC-50" else 10
        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch

        # for logging best so far validation accuracy
        #self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()

    def step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, logits, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, _, preds, targets = self.step(batch)
        # log train metrics
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, logits, preds, targets = self.step(batch)
        metrics = self.metrics(logits, targets)
        self.log_dict(metrics, on_step=False, on_epoch=True)
        return {"loss": loss, "preds": preds, "targets": targets}


    def on_validation_epoch_end(self):
        print("Logging confusion matrix")
        if self.current_epoch % 10 == 0:
            tensorboard = self.logger.experiment
            plt.figure(figsize=(self.cm_figsize, self.cm_figsize))
            ax = sns.heatmap(self.conf_matrix, annot=True, cbar=False, xticklabels=self.classes_list, yticklabels=self.classes_list)
            tensorboard.add_figure(f'Confusion matrix epoch {self.current_epoch}', ax.get_figure())



    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()  # get val accuracy from current epoch
        self.val_acc_best.update(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        acc = self.test_acc(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        self.train_acc.reset()
        self.test_acc.reset()
        self.val_acc.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
