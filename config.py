
from torch import nn
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class DataModule:
  input_root : str                     = "data/patches-DDSM-CBIS"
  image_size : tuple                   = (128,128)
  train_batch_size : int               = 64
  val_batch_size : int                 = 64
  num_workers : int                    = 8
  singleval : bool                     = True




@dataclass
class Train:
  loss:str 				     = 'ContrastiveLoss'
  lr : float				   = 1e-3
  margin:int				   = 10
  num_workers:int 		 = 4
  optimizer : str      = 'sgd'
  scheduler : str      = 'rop'

  classification :bool = True
  segmentation :bool   = True
  singleval :bool      = True

  loss_weights :  tuple = (1,1,1)
  weight_decay : float  = 5e-4

  shared_segmentation : bool = True

@dataclass
class Config:

	datamodule: DataModule
	train: Train
