###
# file name: datawork.py (Wavemotion-lightning)
# Author: Dr. Xiaoxian Guo @ Shanghai Jiao Tong University, China
# Email: xiaoxguo@sjtu.edu.cn
# Website: https://naoce.sjtu.edu.cn/teachers/9004.html
# Github: https://github.com/XiaoxG/waveMotion-lightning/
# Create date: 2021/10/08
# Wavemotion-lightning-- a demonstration code for wave motion prediction of an offshore platform. 
# Refers to preprint Arvix: "Probabilistic prediction of the heave motions of a semi-submersible by a deep learning model"
###

import torch
import os, pickle
from numpy import mean, std, concatenate, random
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from pytorch_lightning import LightningDataModule
import pandas as pd

class Wavedataset(Dataset):
    def __init__(self, wavedata, timestep, forwardstep, wave_lag, motion = 'Surge'):
        self.time_step = timestep
        self.forward_step = forwardstep
        self.wave_lag = wave_lag
        self.data_surge_output = wavedata[motion].values.astype('float32').reshape(-1,1)
        self.data_surge_input = wavedata[motion].values.astype('float32').reshape(-1,1)
        self.data_wave = wavedata['WP'].values.astype('float32').reshape(-1,1)
        self.mean_surge = mean(self.data_surge_output)
        self.mean_wave = mean(self.data_wave)
        self.std_surge = std(self.data_surge_output)
        self.std_wave = std(self.data_wave)

    def __len__(self):
        return len(self.data_wave)-self.time_step-self.forward_step - self.wave_lag +1
    
    def __getitem__(self, idx):
        x = torch.from_numpy(concatenate((self.data_surge_input[idx:idx+self.time_step,:],self.data_wave[idx+self.wave_lag:idx+self.time_step+self.wave_lag,:]),axis=1))
        y = torch.from_numpy(self.data_surge_output[idx+self.time_step:idx+self.time_step+self.forward_step,0])
        return (x, y)

class wavecaseDataModule(LightningDataModule):
    def __init__(self, params):
        super().__init__()
        self.time_step = params.data.time_step
        self.forward_step = params.data.forward_step
        self.wave_lag = params.data.wave_lag
        self.batch_size = params.training.batch_size
        self.motion = params.data.motion
        self.save_data = True
        self.filedic = params.data.filedic
        self.num_workers = params.data.num_workers
        self.train_noise_level = params.data.noise_level

    def _shared_readfile(self, tag):
        dataset = []
        for file in os.listdir(self.filedic+tag):
            if os.path.splitext(file)[1]=='.pickle':
                file_name = os.path.join(self.filedic+tag, file)
                dataset.append(Wavedataset(pd.read_pickle(file_name), self.time_step, self.forward_step, self.wave_lag, motion = self.motion))
        return dataset
    
    def _shared_prepare_data(self, dataset, nor_para, noise_level):
        sd_surge, sd_wave, mean_surge, mean_wave = nor_para
        sigma = sd_surge * noise_level
        for w in dataset:
            n = len(w.data_surge_input)
            noise = random.default_rng().normal(0, sigma, (n,1))
            w.data_surge_input += noise
            w.data_surge_input = (w.data_surge_input - mean_surge)/sd_surge 
            w.data_surge_output = (w.data_surge_output-mean_surge)/sd_surge 
            w.data_wave = (w.data_wave-mean_wave)/sd_wave   
        return ConcatDataset(dataset)

    def prepare_data(self):
        # called only on 1 GPU
        # download data
        trainset = self._shared_readfile('train_data/')
        valset = self._shared_readfile('val_data/')
        testset = self._shared_readfile('test_data/')

        sd_surge = mean([w.std_surge for w in trainset])
        sd_wave = mean([w.std_wave for w in trainset])
        mean_surge = mean([w.mean_surge for w in trainset])
        mean_wave = mean([w.mean_wave for w in trainset])
        nor_para = (sd_surge, sd_wave, mean_surge, mean_wave)

        trainset = self._shared_prepare_data(trainset, nor_para, self.train_noise_level)
        # testset = self._shared_prepare_data(testset, nor_para)        
        valset = self._shared_prepare_data(valset, nor_para, self.train_noise_level)
        
        with open('./train_test_dataset.pickle', 'wb') as output:  # Overwrites any existing file.
            pickle.dump((trainset, valset, testset, nor_para), output, pickle.HIGHEST_PROTOCOL)

    def setup(self):
        # called on every GPU
        with open('./train_test_dataset.pickle', 'rb') as f:
            self.dataset_train, self.dataset_val, self.dataset_test, self.nor_para = pickle.load(f)

    def train_dataloader(self, shuffle=True):
        train_loader = DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=shuffle, drop_last=True, num_workers=self.num_workers, pin_memory=True)
        return train_loader

    def val_dataloader(self, shuffle=False):
        val_loader = DataLoader(self.dataset_val, batch_size=self.batch_size, shuffle=shuffle,drop_last=True, num_workers=self.num_workers, pin_memory=True)
        return val_loader

    def test_dataloader(self, noise_level,shuffle=False):
        testset = self._shared_prepare_data(self.dataset_test, self.nor_para, noise_level)
        test_loader = DataLoader(testset, batch_size=self.batch_size, shuffle=shuffle,drop_last=True, num_workers=self.num_workers, pin_memory=True)
        return test_loader

if __name__ == '__main__':
    pass