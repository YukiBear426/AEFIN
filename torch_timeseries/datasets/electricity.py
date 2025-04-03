import os
import resource
from.dataset import Dataset, TimeSeriesDataset
from typing import Any, Callable, List, Optional
import torch
from torchvision.datasets.utils import download_and_extract_archive, check_integrity, download_url
import pandas as pd
import numpy as np
import torch.utils.data

'''

class Electricity(TimeSeriesDataset):
    """The raw dataset is in https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014. 
    It is the electricity consumption in kWh was recorded every 15 minutes from 2011 to 2014.
    Because the some dimensions are equal to 0. So we eliminate the records in 2011.
    Final we get data contains electircity consumption of 321 clients from 2012 to 2014. 
    And we converted the data to reflect hourly consumption.
    
    due to the missing data , we use the data processed in paper
    《H. Wu, T. Hu, Y. Liu, H. Zhou, J. Wang, and M. Long, “TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis.”》
    
    """
    name:str= 'electricity'
    num_features: int = 321
    freq: str = 't' # in minuts
    length :int = 26304
    
    def download(self):
        download_url(
            "https://raw.githubusercontent.com/wayne155/multivariate_timeseries_datasets/main/electricity/electricity.csv",
            self.dir,
            filename="electricity.csv",
            md5="a1973ba4f4bed84136013ffa1ca27dc8",
        )
        
    def _load(self) -> np.ndarray:
        self.file_name = os.path.join(self.dir, 'electricity.csv')
        self.df = pd.read_csv(self.file_name, parse_dates=['date'])
        self.dates = pd.DataFrame({'date':self.df.date})
        self.data = self.df.drop("date", axis=1).values
        return self.data

'''

class Electricity(TimeSeriesDataset):
    """The raw dataset is in https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014. 
    It is the electricity consumption in kWh was recorded every 15 minutes from 2011 to 2014.
    Because the some dimensions are equal to 0. So we eliminate the records in 2011.
    Final we get data contains electircity consumption of 321 clients from 2012 to 2014. 
    And we converted the data to reflect hourly consumption.
    
    due to the missing data , we use the data processed in paper
    《H. Wu, T. Hu, Y. Liu, H. Zhou, J. Wang, and M. Long, “TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis.”》
    
    """
    name:str= 'electricity'
    num_features: int = 321
    freq: str = 't' # in minuts
    length :int = 26304//4
    
    def download(self):
        download_url(
            "https://raw.githubusercontent.com/wayne155/multivariate_timeseries_datasets/main/electricity/electricity.csv",
            self.dir,
            filename="electricity.csv",
            md5="a1973ba4f4bed84136013ffa1ca27dc8",
        )
        
    def _load(self) -> np.ndarray:
        """Load the dataset and reduce its size by half in the time dimension."""
        # Load the dataset
        self.file_name = os.path.join(self.dir, 'electricity.csv')
        self.df = pd.read_csv(self.file_name, parse_dates=['date'])
        
        # Extract dates and data
        self.dates = pd.DataFrame({'date': self.df.date})
        self.data = self.df.drop("date", axis=1).values
        
        # Reduce the time dimension by taking the first half of the data
        half_length = len(self.data) // 4
        self.data = self.data[:half_length, :]
        self.dates = self.dates.iloc[:half_length]
        
        return self.data
