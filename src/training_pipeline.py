import os
from typing import List, Optional

import dotenv
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import LightningLoggerBase
from datamodules.audio_datamodule import AudioTagDataModule
from models.audiotag_module import AudioTaggerLM
import utils

log = utils.get_logger(__name__)
dotenv.load_dotenv(override=True)


@hydra.main(config_path="../configs/", config_name="train.yaml")
def train(config: DictConfig) -> Optional[float]:
    
    # Set seed for random number generators in pytorch, numpy and python.random
    log.info(f"Seeding RNGs for the run with seed {config.seed}")
    seed_everything(config.seed, workers=True)

    # Init lightning datamodule
    log.info(f"Instantiating datamodule for Dataset <{config.datamodule.dataset}>")
    datamodule: LightningDataModule = AudioTagDataModule(config)

    # Init lightning model
    log.info(f"Instantiating model <{config.Model.Net._target_}>")
    model: LightningModule = AudioTaggerLM(config)

    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(instantiate(cb_conf))

    # Init logger
    logger: LightningLoggerBase = instantiate(config.Logger)
    
    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.Trainer._target_}>")
    trainer: Trainer = instantiate(config.Trainer,
                                   callbacks=callbacks,
                                   logger=logger,
                                   _convert_="partial",)

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters")
    utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Train the model
    log.info("Starting training")
    trainer.fit(model=model, datamodule=datamodule)
        
    # Get metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric and optimized_metric not in trainer.callback_metrics:
        raise Exception(
            "Metric for hyperparameter optimization not found! "
            "Make sure the `optimized_metric` in `hparams_search` config is correct!"
        )
    score = trainer.callback_metrics.get(optimized_metric)
    
    # Print path to best checkpoint
    if not config.trainer.get("fast_dev_run") and config.get("train"):
        log.info(f"Best model ckpt at {trainer.checkpoint_callback.best_model_path}")

    # Return metric score for hyperparameter optimization
    return score


if __name__ == "__main__":
    train()