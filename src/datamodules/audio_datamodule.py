from typing import Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from omegaconf import DictConfig
from hydra.utils import instantiate

class AudioTagDataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)
        # data transformations - instantiated from config


    @property
    def num_classes(self) -> int:
        return self.cfg.datamodule.num_classes

    def prepare_data(self):
        """Download data if needed.

        This method is called only from a single GPU.
        Do not use it to assign state (self.x = y).
        """
        # TODO maybe make one of these for one of the datasets?
        return

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """
        self.train_set = instantiate(self.cfg.datamodule.training_dataset, mode="train", _recursive_=False)
        self.val_set   = instantiate(self.cfg.datamodule.validation_dataset, cfg=self.cfg)
        self.test_set  = instantiate(self.cfg.datamodule.test_dataset, cfg=self.cfg)

    def train_dataloader(self):

        return DataLoader(
            dataset=self.trainset,
            batch_size=self.cfg.datamodule.batch_size,
            num_workers=self.cfg.datamodule.num_workers,
            pin_memory=self.cfg.datamodule.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_set,
            batch_size=self.cfg.datamodule.batch_size,
            num_workers=self.cfg.datamodule.num_workers,
            pin_memory=self.cfg.datamodule.pin_memory,
            shuffle=True,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_set,
            batch_size=self.cfg.datamodule.batch_size,
            num_workers=self.cfg.datamodule.num_workers,
            pin_memory=self.cfg.datamodule.pin_memory,
            shuffle=True,
        )
