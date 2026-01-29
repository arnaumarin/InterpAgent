import numpy as np
import torch
from typing import Tuple
from pathlib import Path
from torch.utils.data import Dataset  

def data_path(train_path=None, test_path=None):
  global train_data_path
  global test_data_path 
  train_data_path = train_path
  test_data_path = test_path
  
class AutosortDataset():
  def __init__(self):
    self.test_x = np.load(Path(test_data_path)/'test_x.npy')
    self.test_y = np.load(Path(test_data_path)/'test_y.npy')
    if train_data_path is not None:
      self.train_x = np.load(Path(train_data_path)/'train_x.npy')
      self.train_y = np.load(Path(train_data_path)/'train_y.npy')
    
class ExpDataset(Dataset):
  def __init__(self, is_train, gain=1, bias=0):
    super().__init__()
    autosort_dset = AutosortDataset() # Already has train/test flattend images.
    self.gain, self.bias = gain, bias # Gain and Bias for encoding.
    if is_train:
      self.features = autosort_dset.train_x
      self.labels = np.int64(autosort_dset.train_y) # Ensure lables are long.
    else:
      self.features = autosort_dset.test_x
      self.labels = np.int64(autosort_dset.test_y) # Ensure labels are long.

  def __len__(self):
    return len(self.labels)

  def __getitem__(self,idx):

    return self.features[idx], self.labels[idx]