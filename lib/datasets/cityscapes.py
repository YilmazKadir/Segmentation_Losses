import logging
import json
import albumentations as A
import numpy as np
from PIL import Image
from albumentations.pytorch import ToTensorV2

from lib.dataset import VoxelizationDataset
from lib.utils import read_txt


class CityScapesDataset(VoxelizationDataset):
    
  # Augmentation arguments
  NUM_IN_CHANNEL = 3
  NUM_LABELS = 34
  IGNORE_LABELS = (0,1,2,3,4,5,6,9,10,14,15,16,18,29,30)

  def __init__(self,
               config,
               augment_data=True,
               phase="train"):
    data_root = config.cityscapes_path
    img_paths = read_txt("splits/cityscapes/" + phase + ".txt")
    seg_paths=[]
    for path in img_paths:
      path = path.replace('leftImg8bit/', 'gtFine/')
      path = path.replace('_leftImg8bit', '_gtFine_labelIds')
      seg_paths.append(path)
    
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
      ignore_index=config.ignore_index,
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
  