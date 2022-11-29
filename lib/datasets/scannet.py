import logging
import os
import numpy as np

from lib.dataset import VoxelizationDataset
from lib.utils import read_txt
import lib.transforms as t

VALID_CLASS_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)


class ScannetVoxelizationDataset(VoxelizationDataset):

  # Voxelization arguments
  VOXEL_SIZE = 0.05
  
  # Augmentation arguments
  NUM_IN_CHANNEL = 3
  COLOR_JITTER_STD = 0.05

  ROTATION_AXIS = 'z'
  NUM_LABELS = 41  # Will be converted to 20 as defined in IGNORE_LABELS.
  IGNORE_LABELS = tuple(set(range(41)) - set(VALID_CLASS_IDS))

  def __init__(self,
               config,
               augment_data=True,
               return_inverse=False,
               merge=False,
               phase="train"):
    data_root = config.scannet_path
    data_paths = read_txt("splits/scannet/" + phase + ".txt")
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
      'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
      'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator',
      'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']
    return classnames

class ScannetVoxelization2cmDataset(ScannetVoxelizationDataset):
  VOXEL_SIZE = 0.02
  