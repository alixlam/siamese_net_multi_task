import os
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split, DataLoader
from typing import Tuple, Optional, Callable, NewType
from data.mammoDataPair import mammoDataPair
from data.mammoDataSingle import mammoDataSingle
from test import val_net_single, val_net_pair



class DataModule(LightningDataModule):

  """ A Lightning Trainer uses a model and a datamodule. Here is defined a datamodule.
        It's basically a wrapper around dataloaders.
    """  
    
  def __init__(self, input_root,image_size, train_batch_size, num_workers, singleval):

    super().__init__()
    self.input_root               = input_root
    self.image_size               = image_size
    self.train_batch_size 		    = train_batch_size
    self.num_workers			        = num_workers
    self.singleval                = singleval


  def train_dataloader(self):
    dataset = mammoDataPair(self.input_root,'train', self.image_size)
    return DataLoader(dataset, self.train_batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)


  def val_dataloader(self):
    if self.singleval == True:
      dataset = mammoDataSingle(self.input_root,'val',self.image_size)
    else :
      dataset = mammoDataPair(self.input_root, 'val', self.image_size)
    return DataLoader(dataset, self.train_batch_size, num_workers = self.num_workers)
  
  def test_dataloader(self):
    dataset = mammoDataSingle(self.input_root, 'test', self.image_size)
    return DataLoader(dataset, self.train_batch_size, num_workers = self.num_workers)

  @classmethod
  def from_config(cls, config):
      """ From a DataModule config object (see config.py) instanciate a Datamodule object. """
      return cls(config.input_root,config.image_size, config.train_batch_size, config.num_workers, config.singleval)