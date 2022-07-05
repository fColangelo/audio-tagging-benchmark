from pytorch_lightning.callbacks import Callback
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig

class Progressive_training(Callback):
    new_res_lr: float 
    adaptation_lr: float                  
    epochs_new_res: int
    opt_cfg: DictConfig
    splits: DictConfig
    
    def __init__(self, new_res_lr: float, adaptation_lr: float, epochs_new_res: int,\
                       optimizer_cfg: DictConfig, splits: DictConfig):
        self.new_res_lr     = new_res_lr
        self.adaptation_lr  = adaptation_lr
        self.epochs_new_res = epochs_new_res
        self.optimizer_cfg  = optimizer_cfg
        self.splits = splits
        

    def on_train_epoch_start(self, trainer, pl_module):
        # Changes optimizer and training dataset to do progressive training
        for idx in range(len(self.splits)): 
            if trainer.current_epoch == self.splits[idx].ep_start:
                # Adjust Dataset
                trainer.datamodule.train_set.adjust_resolution(self.splits[idx].f_res, self.splits[idx].t_res)
                # Reset optimizer to adapt to the new resolution
                new_optim = instantiate(self.optimizer_cfg, params=pl_module.model.parameters(), lr=self.new_res_lr)
                trainer.optimizers = [new_optim]
                
                # TODO implement schedulers
                trainer.lr_schedulers = trainer.configure_schedulers([new_schedulers])
                trainer.optimizer_frequencies = [] # or optimizers frequencies if you have any
            elif trainer.current_epoch == self.splits[idx].ep_end:
                # Reset optimizer after the adaptation phase to the new resolution
                new_optim = instantiate(self.optimizer_cfg, params=pl_module.model.parameters(), lr=self.adaptation_lr)
                trainer.optimizers = [new_optim]

