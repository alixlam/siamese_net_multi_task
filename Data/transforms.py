import numpy as np
from PIL import Image
import random
import math

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomResizedCrop(object):
  def __init__(self,size, scale, ratio, interpolation=Image.BILINEAR):
    super().__init__()
    self.size = size
    self.scale = scale
    self.ratio = ratio 
    self.interpolation = interpolation

  def __call__(self, image, target):
    width, height = F._get_image_size(image)
    area = height * width

    for _ in range(10):
      target_area = area * torch.empty(1).uniform_(self.scale[0], self.scale[1]).item()
      log_ratio = torch.log(torch.tensor(self.ratio))
      aspect_ratio = torch.exp(
        torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
      ).item()

      w = int(round(math.sqrt(target_area * aspect_ratio)))
      h = int(round(math.sqrt(target_area / aspect_ratio)))

    if 0 < w <= width and 0 < h <= height:
      i = torch.randint(0, height - h + 1, size=(1,)).item()
      j = torch.randint(0, width - w + 1, size=(1,)).item()

    in_ratio = float(width) / float(height)
    if in_ratio < min(self.ratio):
      w = width
      h = int(round(w / min(self.ratio)))
    elif in_ratio > max(self.ratio):
      h = height
      w = int(round(h * max(self.ratio)))
    else:  # whole image
      w = width
      h = height
    i = (height - h) // 2
    j = (width - w) // 2


    image = F.resized_crop(image, i, j, h, w, self.size, self.interpolation)
    target = F.resized_crop(target, i,j,h,w, self.size, interpolation=Image.NEAREST)
    return image, target


class RandomHorizontalFlip(object):
  def __init__(self, flip_prob = 0.5):
    self.flip_prob = flip_prob

  def __call__(self, image, target):
    if random.random() < self.flip_prob:
      image = F.hflip(image)
      target = F.hflip(target)
    return image, target



class ToTensor(object):
  def __call__(self, image, target):
    image = F.to_tensor(image)
    target = F.to_tensor(target)
    return image, target


class Resize(object):
  def __init__(self, img_size, interpolation = Image.BILINEAR):
    self.img_size = img_size
    self.interpolation = interpolation
  def __call__(self, image, target):
    image = F.resize(image, self.img_size, self.interpolation)
    target = F.resize(target, self.img_size, interpolation = Image.NEAREST)
    return image, target


class Normalize(object):
  def __init__(self, mean, std):
    self.mean = mean
    self.std = std

  def __call__(self, image, target):
    image = F.normalize(image, mean=self.mean, std=self.std)
    return image, target


class RandomRotation(object):
  def __init__(self, degrees):
    self.degrees = _setup_angle(degrees, name="degrees", req_sizes=(2, ))

  def __call__(self, image, target):
    angle = float(torch.empty(1).uniform_(float(self.degrees[0]), float(self.degrees[1])).item())
    image = F.rotate(image, angle)
    target = F.rotate(target, angle)

    return image, target


def _setup_angle(x, name, req_sizes=(2, )):
    x = [-x, x]

    return [float(d) for d in x]
