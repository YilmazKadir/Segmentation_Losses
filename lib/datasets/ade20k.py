import logging
import json
import albumentations as A
import numpy as np
from PIL import Image
from albumentations.pytorch import ToTensorV2

from lib.dataset import VoxelizationDataset


class ADE20KDataset(VoxelizationDataset):
    
  # Augmentation arguments
  NUM_IN_CHANNEL = 3
  NUM_LABELS = 151
  IGNORE_LABELS = (0,)

  def __init__(self,
               config,
               augment_data=True,
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
      augment_data=augment_data,
      config=config)
    
  def load_data(self, index):
    img_path = self.data_root / self.img_paths[index]
    seg_path = self.data_root / self.seg_paths[index]
    img = Image.open(img_path).convert('RGB')
    seg = Image.open(seg_path)
    img = img.resize((480,360), resample=Image.BILINEAR)
    seg = seg.resize((480,360), resample=Image.NEAREST)
    return np.array(img), np.array(seg)
  
  def get_classnames(self):
    classnames = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
      'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator',
      'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']
    return classnames
  