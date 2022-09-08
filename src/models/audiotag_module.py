from typing import Any, List

import torch
from pytorch_lightning import LightningModule

from torchmetrics import MetricCollection

import matplotlib.pyplot as plt
import seaborn as sns

from omegaconf.dictconfig import DictConfig
from hydra.utils import instantiate

class AudioTaggerLM(LightningModule):

    def __init__(
        self,
        cfg: DictConfig,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.cfg = cfg

        self.net = instantiate(self.cfg.Model.Net)

        # loss function
        self.criterion = instantiate(self.cfg.Model.Loss)
        # Metrics
        self.metrics = MetricCollection(dict(instantiate(self.cfg.Model.Metrics.Metric_list)))

    def forward(self, x: torch.Tensor):
        return self.net(x)

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
        if "Confusion_matrix" in self.metrics:
            log.info("Logging confusion matrix")
            if self.current_epoch % 10 == 0:
                tensorboard = self.logger.experiment
                plt.figure(figsize=(self.cfg.model.Metrics.cm_figsize,
                                    self.cfg.model.Metrics.cm_figsize))
                ax = sns.heatmap(self.conf_matrix,
                                annot=True,
                                cbar=False,
                                xticklabels=self.classes_list,
                                yticklabels=self.classes_list)
                tensorboard.add_figure(f'Confusion matrix epoch {self.current_epoch}', ax.get_figure())






    def configure_optimizers(self):
        optimizer = instantiate(self.cfg.Model.Optimizer, 
                                params=self.net.parameters())
        opt_dict = {"optimizer": optimizer}
        if "Scheduler" in self.cfg.Model:
            scheduler = instantiate(self.cfg.Model.Scheduler, 
                                    optimizer=optimizer)
            opt_dict["lr_scheduler"] = scheduler          
        
        return opt_dict