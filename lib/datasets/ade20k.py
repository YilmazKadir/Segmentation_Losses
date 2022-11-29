import logging
import os
import numpy as np

from lib.dataset import VoxelizationDataset
from lib.utils import read_txt
import lib.transforms as t
from torchvision import transforms
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

class ADE20KDataset(VoxelizationDataset):
    
  # Augmentation arguments
  NUM_IN_CHANNEL = 3
  NUM_LABELS = 151
  IGNORE_LABELS = (0,)

  def __init__(self,
               config,
               augment_data=True,
               return_inverse=False,
               phase="train"):
    data_root = config.ade20k_path
    odgt_filepath = data_root + "/" + phase + ".odgt"
    img_paths = [json.loads(x.rstrip())["fpath_img"] for x in open(odgt_filepath, 'r')]
    seg_paths = [json.loads(x.rstrip())["fpath_segm"] for x in open(odgt_filepath, 'r')]
    logging.info('Loading {}: {}'.format(self.__class__.__name__, phase))

    if augment_data:
      augmentations = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RGBShift(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(),
        ToTensorV2() ,
      ])
    else:
      augmentations = A.Compose([
        A.Normalize(),
        ToTensorV2(),
      ])
    
    super().__init__(
      img_paths,
      seg_paths,
      data_root=data_root,
      augmentations=augmentations,
      ignore_mask=config.ignore_mask,
      return_inverse=return_inverse,
      augment_data=augment_data,
      config=config)
    
  def get_classnames(self):
    classnames = [
      'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
      'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator',
      'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']
    return classnames
  