
import os
import resource
from.dataset import Dataset, TimeSeriesDataset
from typing import Any, Callable, List, Optional
import torch
from torchvision.datasets.utils import download_url, download_and_extract_archive, check_integrity
import pandas as pd
import numpy as np
import torch.utils.data


class Weather(TimeSeriesDataset):
    name:str= 'weather'
    num_features: int = 21
    sample_rate:int # in munites
    length : int  = 52696
    freq:str=  'h'
    windows : int = 168
    
    
    def download(self):
        download_and_extract_archive(
         "https://raw.githubusercontent.com/wayne155/multivariate_timeseries_datasets/main/weather.zip",
         self.dir,
         filename='weather.zip',
         md5="fe40dc5c7e1787ad8ae63e2cadb5abfe"
        )
        
        
        
    def _load(self) -> np.ndarray:
        self.file_path =  os.path.join( os.path.join(self.dir, 'weather')  , 'weather.csv') 
        self.df = pd.read_csv(self.file_path,parse_dates=[0])
        self.dates = pd.DataFrame({'date':self.df.date})
        self.data = self.df.iloc[:, 1:].to_numpy()
        return self.data

