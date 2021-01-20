

import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
import pytorch_lightning as pl
from typing import Tuple, Dict
from utils import init_optimizer, init_scheduler
from models import Model
import os, sys, argparse, copy
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
import torch.nn.functional as F
from test import val_net_single, val_net_pair
from losses import ContrastiveLoss, DiceLoss, Weighted_BCE, LossBinary
cudnn.benchmark = True
from data.mammoDataPair import mammoDataPair
from data.mammoDataSingle import mammoDataSingle


class LightningModel(pl.LightningModule):

  def __init__(self, **kwargs) -> None:
    """ Instanciate a Lightning Model. 
    """
    super().__init__()
    self.save_hyperparameters()
    self.net  = Model()
    #.from_config(self.hparams)

  def forward(self, x, y = None):
    return self.net(x, y)

  def configure_optimizers(self):
    """ Instanciate an optimizer and a learning rate scheduler to be used during training.
    Returns:
      Dict: Dict containing the optimizer(s) and learning rate scheduler(s) to be used by
          a Trainer object using this model. 
          The 'monitor' key may be used by some schedulers (e.g: ReduceLROnPlateau).                        
    """
    optimizer = init_optimizer(self.net, self.hparams)
    scheduler = init_scheduler(optimizer, self.hparams)
    return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

  def compute_loss(self,weight,predict_match_cc, predict_match_mlo, predict_class_cc,predict_class_mlo, predict_segm_cc,predict_segm_mlo, true_match, true_class_cc,true_class_mlo , true_segm_cc, true_segm_mlo):
    """ Compute the loss of siamese network (two input images)
    Returns : 
      Dict: Dict containing the total loss (sum of the loss of the three tasks), and each loss correspondin
          to the different tasks
    """
    cls_criterion = nn.CrossEntropyLoss()
    mat_criterion = ContrastiveLoss(margin=self.hparams.margin)
    segm_criterion = DiceLoss()
    weight = torch.FloatTensor([weight]).cuda()
    loss_cls_cc = cls_criterion(predict_class_cc, true_class_cc)
    loss_cls_mlo = cls_criterion(predict_class_mlo, true_class_mlo)

    loss_segm_cc = segm_criterion(predict_segm_cc, true_segm_cc)
    loss_segm_mlo = segm_criterion(predict_segm_mlo, true_segm_mlo)

    loss_mat = mat_criterion(predict_match_cc, predict_match_mlo, true_match)

    weight_clas = self.hparams.loss_weights[0]
    weight_segm = self.hparams.loss_weights[1]
    weight_mat = self.hparams.loss_weights[2]
    tot_loss = self.hparams.classification*(weight_clas*(loss_cls_cc + loss_cls_mlo) + weight_mat*(loss_mat)) + self.hparams.segmentation*weight_segm*(loss_segm_cc+loss_cls_mlo)
    return {'Total loss': tot_loss, 'Classification loss': weight_clas*(loss_cls_cc + loss_cls_mlo), 'Matching loss':weight_mat*(loss_mat), 'Segmentation loss': weight_segm*(loss_segm_cc+loss_cls_mlo)}
  
  def compute_val_loss(self,output_clas, output_segm, clas, mask):
    """Compute the loss when only one image is given
    Returns : sum of tasks loss
    """
    cls_criterion = nn.CrossEntropyLoss()
    segm_criterion = DiceLoss()

    loss_cls = cls_criterion(output_clas, clas)
    loss_segm = segm_criterion(output_segm, mask)

    return self.hparams.classification*self.hparams.loss_weights[0]*loss_cls + self.hparams.segmentation * self.hparams.loss_weights[1]*loss_segm
  @staticmethod
  def dice_score(outputs, targets, ratio=0.5):
    outs = outputs.cpu().detach().numpy()
    targs = targets.cpu().detach().numpy()
    outs = outs.flatten()
    targs = targs.flatten()
    outs[outs > ratio] = np.float32(1)
    outs[outs < ratio] = np.float32(0)    
    return float(2 * (targs * outs).sum())/float(targs.sum() + outs.sum())


  def training_step(self, batch, batch_idx):
    img_cc, img_mlo, gt_cls_cc, gt_cls_mlo, gt_mat, mask_cc, mask_mlo = batch
    img_cc, img_mlo, gt_cls_cc, gt_cls_mlo, gt_mat, mask_cc, mask_mlo = Variable(img_cc.cuda()), Variable(img_mlo.cuda()), Variable(gt_cls_cc.squeeze(1).cuda()), Variable(gt_cls_mlo.squeeze(1).cuda()), Variable(gt_mat.squeeze(1).cuda()), Variable(mask_cc.cuda()), Variable(mask_mlo.cuda())
    cls_cc, fea_cc, segm_cc, cls_mlo, fea_mlo, segm_mlo = self(img_cc, img_mlo)

    weight = 1.
    loss_ = self.compute_loss(weight,fea_cc, fea_mlo, cls_cc, cls_mlo, segm_cc, segm_mlo, gt_mat, gt_cls_cc, gt_cls_mlo, mask_cc, mask_mlo)

    loss = loss_['Total loss']
    clas_loss = loss_['Classification loss']
    segm_loss = loss_['Segmentation loss']
    mat_loss = loss_['Matching loss']

    cls_cc, cls_mlo = cls_cc.squeeze(0).cpu(), cls_mlo.squeeze(0).cpu()
    gt_cls_cc, gt_cls_mlo = gt_cls_cc.cpu(), gt_cls_mlo.cpu()
    
    accuracy = pl.metrics.Accuracy()
    acc = (accuracy(cls_cc, gt_cls_cc)+accuracy(cls_mlo, gt_cls_mlo))/2

    dice = (self.dice_score(segm_cc, mask_cc)+self.dice_score(segm_mlo, mask_mlo))/2
    self.log('Dice Score/Train', dice)
    self.log('Accuracy/Train', acc)
    self.log('Loss/Train', loss)
    self.log('Training Task Loss/Segmentation', segm_loss)
    self.log('Training Task Loss/Classification', clas_loss)
    self.log('Training Task Loss/Matching', mat_loss)
    return {'loss': loss, 'dice': dice}

  def validation_step(self, batch, batch_idx):
    accuracy = pl.metrics.Accuracy()
    if self.hparams.singleval == True:
      img, clas,  mask, files = batch
      img, clas, mask = Variable(img.cuda()), Variable(clas.squeeze(1).cuda()), Variable(mask.cuda())
      output_clas, output_segm = self(img)
      weight = 1.
      dice = self.dice_score(output_segm,mask)
      loss = self.compute_val_loss(output_clas, output_segm,clas, mask)
      output_clas = output_clas.squeeze(0).cpu()
      clas = clas.cpu()
      acc = accuracy(output_clas, clas)

    else :
      img_cc, img_mlo, gt_cls_cc, gt_cls_mlo, gt_mat, mask_cc, mask_mlo = batch
      img_cc, img_mlo, gt_cls_cc, gt_cls_mlo, gt_mat, mask_cc, mask_mlo = Variable(img_cc.cuda()), Variable(img_mlo.cuda()), Variable(gt_cls_cc.squeeze(1).cuda()), Variable(gt_cls_mlo.squeeze(1).cuda()), Variable(gt_mat.squeeze(1).cuda()), Variable(mask_cc.cuda()), Variable(mask_mlo.cuda())
      cls_cc, fea_cc, segm_cc, cls_mlo, fea_mlo, segm_mlo = self(img_cc, img_mlo)

      weight = 1.
      loss_ = self.compute_loss(weight,fea_cc, fea_mlo, cls_cc, cls_mlo, segm_cc, segm_mlo, gt_mat, gt_cls_cc, gt_cls_mlo, mask_cc, mask_mlo)

      loss = loss_['Total loss']
      cls_cc, cls_mlo = cls_cc.squeeze(0).cpu(), cls_mlo.squeeze(0).cpu()

      gt_cls_cc, gt_cls_mlo = gt_cls_cc.cpu(), gt_cls_mlo.cpu()
      acc = (accuracy(cls_cc, gt_cls_cc)+accuracy(cls_mlo, gt_cls_mlo))/2
      dice = (self.dice_score(segm_cc, mask_cc)+self.dice_score(segm_mlo, mask_mlo))/2


    self.log('Dice Score/Validation', dice)
    self.log('Loss/Validation', loss)
    self.log('Accuracy/Validation', acc)
    return {'val_loss': loss,'dice': dice, 'Accuracy': acc}


  def test_step(self, batch, batch_idx):
    accuracy = pl.metrics.Accuracy()
    img, clas,  mask, files = batch
    img, clas, mask = Variable(img.cuda()), Variable(clas.squeeze(1).cuda()), Variable(mask.cuda())
    output_clas, output_segm = self(img)
    weight = 1.
    dice = self.dice_score(output_segm,mask)
    output_clas = output_clas.squeeze(0).cpu()
    clas = clas.cpu()
    acc = accuracy(output_clas, clas)
    self.log('Test/dice' , dice)
    self.log('Test/accuracy' , acc)

  @classmethod
  def from_config(cls, config):
    return cls(
      input_root = config.datamodule.input_root,
      train_batch_size = config.datamodule.train_batch_size,
      num_workers = config.datamodule.num_workers,
      loss = config.train.loss,
      lr = config.train.lr,
      margin = config.train.margin,
      optimizer = config.train.optimizer,
      scheduler = config.train.scheduler,
      weight_decay = config.train.weight_decay,
      classification = config.train.classification,
      segmentation = config.train.segmentation,
      singleval = config.train.singleval,
      loss_weights = config.train.loss_weights
    )