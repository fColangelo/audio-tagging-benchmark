import csv
from glob import glob
import os
import pickle
from typing import Callable, Dict, Tuple

from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig
import torch
from torch.utils.data import Dataset
import torchaudio
from torchaudio.transforms import MelSpectrogram


class General_Dataset(Dataset):
    
    def __init__(self, 
                 audio_cfg: DictConfig,
                 augum_cfg: DictConfig,
                 data_dir:str,
                 file_list: list,
                 dataset_name: str="default name",
                 mode: str="train") -> None:
        super().__init__()
        assert mode in ("train", "val", "test"),\
            f"mode should be one train, val or test"
        
        self.audio_cfg = audio_cfg
        self.augum_cfg = augum_cfg

        self.dataset_name = dataset_name  #Each dataset overrides this
        self.spectrogram = self.get_spectrogram(self.audio_cfg)
        

        #instantiate(self.audio_cfg.transform)
        # Iterable containing all audio file in this split of the dataset
        # Individual dataset have their own method to generate it
        # and pass it to super.init
        self.audio_list = file_list
        # Transforms instantiation from cfg (if defined)
        # ATM, torchaudio is supported
        # Time domain
        self.td_transforms = None
        if self.augum_cfg.time_domain:
            self.td_transforms = []
            for t in self.augum_cfg.time_domain:
                self.td_transforms.append(instantiate(t))
        # Frequency domain
        self.fd_transforms = None
        if self.augum_cfg.frequency_domain:
            self.fd_transforms = []
            for t in self.augum_cfg.frequency_domain:
                self.fd_transforms.append(instantiate(t))
        # Normalization file loading (if it already exists) or calculation
        statfile_name = (f"{self.dataset_name}_"
                         f"{audio_cfg.sample_rate}_"
                         f"{audio_cfg.n_fft}_"
                         f"{audio_cfg.hop_length}_"
                         f"{audio_cfg.n_mels}") 
        statfile_path = f"{data_dir}/{self.dataset_name}/{statfile_name}"
        if os.path.isfile(statfile_path):
            with open(statfile_path, "rb") as f:
                data_dict = pickle.load(f)
                self.mean = data_dict["mean"] 
                self.std  = data_dict["std"]
        elif mode=="train":    
            data_dict = self.get_stats()
            with open(statfile_path, "wb") as f:
                pickle.dump(data_dict, f)
        else:
            raise ValueError("Val dataset created without a train stats file"
                             " available. Create a stat file by instantiating"
                             " a training dataset first")

    def get_spectrogram(self,
                        audio_cfg: DictConfig) -> Callable:
        S = MelSpectrogram(audio_cfg.sample_rate,
                           audio_cfg.n_fft,
                           audio_cfg.win_length,
                           audio_cfg.hop_length,
                           audio_cfg.f_min,
                           audio_cfg.f_max,
                           audio_cfg.n_mels,
                           window_fn=torch.hann_window) 
        return S   
        
    def get_stats(self) -> Dict:
        """Calculates mean and standard deviation for the dataset.
           A single set of stats is returned, as the spectrogram 
           is assumed to be mono-channel. 

        Returns:
            dict: dict containing "mean": meanval, "std": stdval
        """
        mean    = torch.tensor(0)
        sq_mean = torch.tensor(0)
        idx = 0
        for ap in self.audio_list:
            wf, _   = torchaudio.load(ap)
            mean    += torch.mean(wf) 
            sq_mean += torch.mean(torch.square(wf))
            idx     += 1
        mean    /= idx
        sq_mean /= idx
        std      = np.sqrt(sq_mean - mean**2)
        return {"mean": mean, "std": std}
      
    def pad_spectrum(self, 
                     spec: torch.Tensor)-> torch.Tensor:
        while spec.shape[-1] < self.audio_cfg.spec_time_steps:
            spec = torch.cat([spec, spec], dim=-1)
        spec = spec[..., :self.audio_cfg.spec_time_steps]
        return spec  
    
    def get_log_spectrum(self, 
                         audio: torch.Tensor) -> torch.Tensor:
        """Transform an audio waveform into a melspectrogram.
           Parameters are defined in cfg.Audio

        Args:
            audio (torch.tensor): audio waveform as read 
            by torchaudio with shape (n_samples)

        Returns:
            torch.tensor: log-melspectrogram of the audio waveform,
            with shape (1, n_mel, t_steps)
        """
        spec = torch.log10(self.spectrogram(audio))
        spec = self.pad_spectrum(spec)       
        return spec

    def normalize(self, 
                  spect: torch.Tensor) -> torch.Tensor:
        spect -= self.mean
        spect /= self.std
        return spect        
         
    def get_label(self,
                  filepath: str) -> None:
        """Abstract class, must be overloaded in specific dataset classes
        """
        return None 
  
    def __len__(self) -> int:
        return len(self.audio_list)
    
    def __getitem__(self, index: int) -> Tuple:
        audio, _ = torchaudio.load(self.audio_list[index])
        label    = self.get_label(self.audio_list[index])
        if self.td_transforms is not None:
            for t in self.td_transforms:
                audio = t(audio)
        spect    = self.get_log_spectrum(audio)
        n_spect  = self.normalize(spect)
        if self.fd_transforms is not None:
            for t in self.fd_transforms:
                n_spect = t(n_spect)
        return n_spect, label   
    
class FSD50K(General_Dataset):    

    def __init__(self,
                 audio_cfg: DictConfig,
                 augum_cfg: DictConfig,
                 data_dir:str,
                 mode: str ="train"):
        dataset_name = "FSD50K"
        file_list    = self.get_file_list(mode, audio_cfg, data_dir)
        self.labels_dict, self.vocab_dict = self.get_label_infos(mode, data_dir)
        super().__init__(audio_cfg,
                         augum_cfg,
                         data_dir,
                         file_list,
                         dataset_name,
                         mode)

    def get_file_list(self,
                      mode: str, 
                      audio_cfg: DictConfig,
                      data_dir:str) -> list:
        """Enumerates all files contained in this split of the dataset

        Args:
            mode (str): Dataset mode to choose the correct label file
            audio_cfg (DictConfig): cfg file, since files are in different
                                    subdir based on sampling 
            data_dir (str): Directory containing dataset files

        Returns:
            list: contains list of filepaths
        """
        folder_name = "eval_audio" if mode=="test" else "dev_audio" 
        file_path   = f"{data_dir}{folder_name}/{audio_cfg.sample_rate}"
        return glob(f"{file_path}*.wav")

    def get_label_infos(self,
                        mode: str,
                        data_dir:str) -> tuple:
        """Read CSVs with sample: label mapping and label vocabulary

        Args:
            mode (str): Dataset mode to choose the correct label file
            data_dir (str): Directory containing dataset files

        Returns:
            tuple: contains label dict and vocabulary dict
        """
        # Eval contains only test file, dev contains train+val
        labels_filename = "eval.csv" if mode=="test" else "dev.csv"    
        labels_filepath = os.path.join(data_dir, "FSD50K.ground_truth", labels_filename)
        vocab_filepath  = os.path.join(data_dir, "FSD50K.ground_truth", "vocabulary.csv")

        with open(labels_filepath, mode='r') as f:
            reader = csv.reader(f)
            labels_dict = {rows[0]:rows[1] for rows in reader}
            
        with open(vocab_filepath, mode='r') as f:
            reader = csv.reader(f)
            vocab_dict = {rows[1]:rows[0] for rows in reader}
        
        return labels_dict, vocab_dict
    
    def get_label(self, filepath: str) -> torch.Tensor:
        """Obtain a tensor encoding the label from the filepath name

        Args:
            filepath (str): filepath of the audio file

        Returns:
            torch.Tensor: one-hot encoded label tensor 
        """
        filename   = filepath.split("/")[-1].split(".")[0]
        str_labels = self.labels_dict[filename]
        str_label_list = str_labels.split(",")
        int_label_list = [self.vocab_dict[x] for x in str_label_list]
        label = torch.zeros(len(self.vocab_dict))
        label[int_label_list] = 1
        return label 