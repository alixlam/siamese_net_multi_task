import os
import os.path
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from tqdm import tqdm
from data.transforms import Compose, RandomRotation, RandomHorizontalFlip, RandomResizedCrop, Resize, ToTensor, Normalize


class mammoDataSingle(Dataset):
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
        self.image_list = list()
        for file_name_cc in os.listdir(os.path.join(self.root, self.image_set, 'CC')):
            self.image_list.append(os.path.join(self.root, self.image_set, 'CC', file_name_cc))
        for file_name_mlo in os.listdir(os.path.join(self.root, self.image_set, 'MLO')):
            self.image_list.append(os.path.join(self.root, self.image_set, 'MLO', file_name_mlo))


    def __getitem__(self, index):
        file_name = self.image_list[index]

        file_name = file_name.replace(' (1)','') if ' (1)' in file_name else file_name
        img = Image.open(file_name).convert('RGB')

        file_name = file_name.replace(' (2)','') if ' (2)' in file_name else file_name
        img = Image.open(file_name).convert('RGB')

        file_name_mask = file_name.replace('.png', '-mask.png') if 'neg' not in file_name else file_name.replace('neg-','').replace('.png', '-mask.png')
        
        if self.image_set == 'val':
          file_name_mask = file_name_mask.replace('/CC/', '/mask_val/') if 'CC' in file_name_mask else file_name_mask.replace('/MLO/','/mask_val/' )
        elif self.image_set == 'test':
          file_name_mask = file_name_mask.replace('/CC/', '/mask_test/') if 'CC' in file_name_mask else file_name_mask.replace('/MLO/','/mask_test/' )


        try :
          mask = Image.open(file_name_mask)
        except FileNotFoundError :
          g = ['-00-', '-01-', '-02-', '-03-', '-04-']
          for i, er in enumerate(['-10-','-11-', '-12-', '-13-', '-14-']):
            file_name_mask = file_name_mask.replace(g[i], er) if g[i] in file_name_mask else file_name_mask          
          mask = Image.open(file_name_mask)

        #print('{} -> {}'.format(file_name_mask,file_name))

        if self.preproc is not None:
            img, mask = self.preproc(img,mask)


        gt = torch.Tensor([0]) if 'neg' in file_name else torch.Tensor([1])
        
        return img, gt.type(torch.LongTensor), mask, (file_name, file_name_mask)
        #,file_name

    def __len__(self):
        return len(self.image_list)

if __name__ == '__main__':
    i = mammoDataSingle('/data/patches-DDSM-CBIS/','train')
    print(len(i))