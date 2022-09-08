from typing import Optional, Tuple

from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader


class AudioTagDataModule(LightningDataModule):

    def __init__(self,
                 cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # this line allows to access init params 
        # with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)
        # data transformations - instantiated from config

    @property
    def num_classes(self) -> int:
        return self.cfg.datamodule.num_classes

    def setup(self, 
              stage: Optional[str] = None) -> None:
        """
        This method is called by lightning when doing `trainer.fit()`
        and `trainer.test()`, so be careful not to execute 
        the random split twice! The `stage` can be used to
        differentiate whether it's called before 
        trainer.fit()` or `trainer.test()`.
        """
        self.train_set = instantiate(self.cfg.datamodule.training_dataset,
                                     mode="train",
                                     _recursive_=False)
        self.val_set   = instantiate(self.cfg.datamodule.validation_dataset,
                                     cfg=self.cfg)
        self.test_set  = instantiate(self.cfg.datamodule.test_dataset,
                                     cfg=self.cfg)
    
    def train_dataloader(self) -> DataLoader:

        return DataLoader(
            dataset=self.train_set,
            batch_size=self.cfg.datamodule.batch_size,
            num_workers=self.cfg.datamodule.num_workers,
            pin_memory=self.cfg.datamodule.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_set,
            batch_size=self.cfg.datamodule.batch_size,
            num_workers=self.cfg.datamodule.num_workers,
            pin_memory=self.cfg.datamodule.pin_memory,
            shuffle=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_set,
            batch_size=self.cfg.datamodule.batch_size,
            num_workers=self.cfg.datamodule.num_workers,
            pin_memory=self.cfg.datamodule.pin_memory,
            shuffle=True,
        )
