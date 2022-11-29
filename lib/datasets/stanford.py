import logging
import os
import numpy as np

from lib.utils import read_txt
from lib.dataset import VoxelizationDataset
import lib.transforms as t

CLASSES = [
    'clutter', 'beam', 'board', 'bookcase', 'ceiling', 'chair', 'column', 'door', 'floor', 'sofa',
    'table', 'wall', 'window'
]


class StanfordArea5Dataset(VoxelizationDataset):

  # Voxelization arguments
  VOXEL_SIZE = 0.05  # 5cm
    
  # Augmentation arguments
  NUM_IN_CHANNEL = 3
  COLOR_TRANS_RATIO = 0.05
  COLOR_JITTER_STD = 0.005
  
  ROTATION_AXIS = 'z'
  NUM_LABELS = 13
  IGNORE_LABELS = ()
  
  def __init__(self,
               config,
               augment_data=True,
               return_inverse=False,
               merge=False,
               phase="train"):
    data_root = config.stanford3d_path
    data_paths = read_txt("splits/stanford/" + phase )
    logging.info('Loading {}: {}'.format(self.__class__.__name__, phase))
    
    if augment_data:
      augmentations = t.Compose([
        t.RandomTranslateRotateScale(),
        t.ElasticDistortion(),
        t.RandomHorizontalFlip(self.ROTATION_AXIS, self.IS_TEMPORAL),
        t.RandomBrightnessContrast(),
        t.RGBShift()
      ])
    else:
      augmentations = None
        
    super().__init__(
      data_paths,
      data_root=data_root,
      augmentations=augmentations,
      ignore_mask=config.ignore_mask,
      return_inverse=return_inverse,
      augment_data=augment_data,
      config=config,
      merge=merge)
  
  def get_classnames(self):
    classnames = [
      'clutter', 'beam', 'board', 'bookcase', 'ceiling', 'chair', 'column', 'door', 'floor',
      'sofa', 'table', 'wall', 'window']
    return classnames