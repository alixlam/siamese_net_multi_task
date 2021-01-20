import os
import os.path
import random
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from tqdm import tqdm
from data.transforms import Compose, RandomRotation, RandomHorizontalFlip, RandomResizedCrop, Resize, ToTensor, Normalize

class mammoDataPair(Dataset):
    def __init__(self, root, image_set,image_size):
        """
        root: path
        image_set: train/val/test
        root/train/CC/...
        root/train/MLO/...
        root/val/CC/...
        root/val/MLO/...
        root/test/CC/...
        root/test/MLO/...
        """
        self.root = root
        self.image_set = image_set
        self.image_size=image_size
        
        data_transforms = {
            'train': Compose([
                RandomRotation(25),
                RandomHorizontalFlip(),
                RandomResizedCrop(self.image_size, scale=(0.96, 1.0), ratio=(0.95, 1.05)),
                Resize(self.image_size),
                ToTensor(),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': Compose([
                Resize(self.image_size),
                ToTensor(),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': Compose([
                Resize(self.image_size),
                ToTensor(),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'all': Compose([
                Resize(self.image_size),
                ToTensor(),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
  

        self.preproc = data_transforms[self.image_set]

        self.cc_list, self.mlo_list = list(), list()
        for file_name_cc in os.listdir(os.path.join(self.root, self.image_set, 'CC')):
            img_id = file_name_cc.split('-')[0]
            for file_name_mlo in os.listdir(os.path.join(self.root, self.image_set, 'MLO')):
                if img_id in file_name_mlo:
                    if ('RCC' in file_name_cc) and ('LMLO' in file_name_mlo):
                        continue
                    if ('LCC' in file_name_cc) and ('RMLO' in file_name_mlo):
                        continue
                    self.cc_list.append(os.path.join(self.root, self.image_set, 'CC', file_name_cc))
                    self.mlo_list.append(os.path.join(self.root, self.image_set, 'MLO', file_name_mlo))
                    #print('{} ==> {}'.format(file_name_cc,file_name_mlo))
        

    def __getitem__(self, index):
        file_name_cc = self.cc_list[index]
        file_name_mlo = self.mlo_list[index]

        file_name_cc = file_name_cc.replace(' (1)', '') if ' (1)' in file_name_cc else file_name_cc
        file_name_mlo = file_name_mlo.replace(' (1)', '') if ' (1)' in file_name_mlo else file_name_mlo

        file_name_cc = file_name_cc.replace(' (2)', '') if ' (2)' in file_name_cc else file_name_cc
        file_name_mlo = file_name_mlo.replace(' (2)', '') if ' (2)' in file_name_mlo else file_name_mlo

        file_name_cc_mask = file_name_cc.replace('.png', '-mask.png') if 'neg' not in file_name_cc else file_name_cc.replace('neg-','').replace('.png', '-mask.png')
        file_name_mlo_mask = file_name_mlo.replace('.png', '-mask.png') if 'neg' not in file_name_mlo else file_name_mlo.replace('neg-','').replace('.png', '-mask.png')
        
        file_name_cc_mask = file_name_cc_mask.replace('/CC/', '/mask_'+self.image_set+'/')
        file_name_mlo_mask = file_name_mlo_mask.replace('/MLO/', '/mask_'+self.image_set+'/')

        img_cc = Image.open(file_name_cc).convert('RGB')
        img_mlo =Image.open(file_name_mlo).convert('RGB')

        try :
          mask_cc = Image.open(file_name_cc_mask)
        except FileNotFoundError :
          g = ['-00-', '-01-', '-02-', '-03-', '-04-']
          for i, er in enumerate(['-10-','-11-', '-12-', '-13-', '-14-']):
            file_name_cc_mask = file_name_cc_mask.replace(g[i], er) if g[i] in file_name_cc_mask else file_name_cc_mask          
          mask_cc = Image.open(file_name_cc_mask)

        try :
          mask_mlo = Image.open(file_name_mlo_mask)
        except FileNotFoundError :
          g = ['-00-', '-01-', '-02-', '-03-', '-04-']
          for i, er in enumerate(['-10-','-11-', '-12-', '-13-', '-14-']):
            file_name_mlo_mask = file_name_mlo_mask.replace(g[i], er) if g[i] in file_name_mlo_mask else file_name_mlo_mask          
          mask_mlo = Image.open(file_name_mlo_mask)


        if self.preproc is not None:
            img_cc, mask_cc = self.preproc(img_cc, mask_cc)
            img_mlo, mask_mlo = self.preproc(img_mlo, mask_mlo)

        gt_cls_cc = torch.Tensor([0]) if 'neg' in file_name_cc else torch.Tensor([1])
        gt_cls_mlo = torch.Tensor([0]) if 'neg' in file_name_mlo else torch.Tensor([1])
        if not('neg' in file_name_cc) and not('neg' in file_name_mlo):
            gt_mat = torch.Tensor([1])
        elif ('neg' in file_name_cc) and ('neg' in file_name_mlo):
            gt_mat = torch.Tensor([-1])
        else:
            gt_mat = torch.Tensor([0])
                
        return img_cc, img_mlo, gt_cls_cc.type(torch.LongTensor), gt_cls_mlo.type(torch.LongTensor), gt_mat.type(torch.LongTensor), mask_cc,mask_mlo

    def __len__(self):
        assert len(self.cc_list) == len(self.mlo_list)
        return len(self.cc_list)

if __name__ == '__main__':
    i = mammoDataPair('/data/patches-DDSM-CBIS/','train')
    print(len(i))