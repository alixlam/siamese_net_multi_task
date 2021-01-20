import os
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from lightningModel import LightningModel
from data.datamodule import DataModule
import config as cfg


def init_trainer():
  lr_logger      = LearningRateMonitor()
  early_stopping = EarlyStopping(monitor   = 'Dice Score/Validation',
                                 mode      = 'min', 
                                 min_delta = 0.001,
                                 patience  = 100,
                                 verbose   = True)
  return Trainer(gpus=1, callbacks = [lr_logger, early_stopping])


def run_colab_training(config):
  print('Instancing model...')
  data = DataModule.from_config(config.datamodule)
  model = LightningModel.from_config(config)
  try: 
    trainer = init_trainer()
    print('Ready. Training will start !')
    trainer.fit(model, data)
  except MisconfigurationException:
    print('Did you forget to setup a GPU runtime ?  if not then the error might be in your configuration check config file')
  trainer.fit(model, data)


def download_outputs(file_module):
  os.system('zip -r /content/output.zip /content/projectS5/lightning_logs/version_0/')
  file_module.download("/content/output.zip")

'''if __name__ == '__main__':
    config_train = cfg.DataModule()
    config_datamodule = cfg.Train()
    config = cfg.Config(config_datamodule, config_train)
    run_colab_training(config)'''
