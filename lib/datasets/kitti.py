import logging
import os
import numpy as np
from collections import defaultdict

from lib.dataset import TemporalVoxelizationDataset, VoxelizationDataset
from lib.utils import read_txt
import lib.transforms as t


class KITTIVoxelizationDataset(VoxelizationDataset):

  # Voxelization arguments
  VOXEL_SIZE = 0.05
  
  # Augmentation arguments
  NUM_IN_CHANNEL = 2
  
  ROTATION_AXIS = 'z'
  NUM_LABELS = 20  # Will be converted to 19 as defined in IGNORE_LABELS.
  IGNORE_LABELS = (0,)
  
  def __init__(self,
               config,
               augment_data=True,
               return_inverse=False,
               merge=False,
               phase="train"):
    data_root = config.kitti_path
    data_paths = read_txt("splits/kitti/" + phase + ".txt" )
    logging.info('Loading {}: {}'.format(self.__class__.__name__, phase))

    if augment_data:
      augmentations = t.Compose([
        t.RandomTranslateRotateScale(translation_aug_prob=0.5),
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

  def load_npy(self, index):
    filepath = self.data_root / self.data_paths[index]
    points = np.load(self.data_root / filepath)
    coords, feats, labels = (
      points[:, :3],
      points[:, 3:-1],
      points[:, -1],
    )
    # Moving objects to objects for 3D segmentations
    labels[labels == 20] = 1
    labels[labels == 21] = 7
    labels[labels == 22] = 6
    labels[labels == 23] = 8
    labels[labels == 24] = 5
    labels[labels == 25] = 4
    
    return coords, feats, labels
  
  def get_classnames(self):
    classnames = [
      'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person', 'bicyclist', 'motorcyclist', 'road',
      'parking', 'sidewalk', 'other-ground', 'building', 'fence', 'vegetation', 'trunk', 'terrain', 'pole',
      'traffic-sign'
    ]
    return classnames
  
  

class TemporalKITTIVoxelizationDataset(TemporalVoxelizationDataset):
  IS_TEMPORAL = True
  
  # Voxelization arguments
  VOXEL_SIZE = 0.05
    
  # Augmentation arguments
  NUM_IN_CHANNEL = 4
  
  ROTATION_AXIS = 'z'
  NUM_LABELS = 26
  IGNORE_LABELS = (0,)

  def __init__(self,
               config,
               augment_data=True,
               return_inverse=False,
               merge=False,
               phase="train"):
    data_root = config.kitti_path
    data_paths = read_txt("splits/kitti/" + phase + ".txt" )
    self.poses = np.load(data_root + "/poses.npy", allow_pickle=True)
    seq2files = defaultdict(list)
    for f in data_paths:
      seq_name = f[-13:-11]
      seq2files[seq_name].append(f)
    file_seq_list = []
    for key in sorted(seq2files.keys()):
      file_seq_list.append(sorted(seq2files[key]))
    logging.info('Loading {}: {}'.format(self.__class__.__name__, phase))

    if augment_data:
      augmentations = t.Compose([
        t.RandomTranslateRotateScale(translation_aug_prob=0.5, is_temporal=True),
      ])
    else:
      augmentations = None
        
    super().__init__(
      file_seq_list,
      data_root=data_root,
      augmentations=augmentations,
      augment_data=augment_data,
      ignore_mask=config.ignore_mask,
      return_inverse=return_inverse,
      config=config,
      merge=merge,
      temporal_dilation=config.temporal_dilation,
      temporal_numseq=config.temporal_numseq)
  
  def load_world_pointcloud(self, seq_idx, sweep_idx):
    filepath = self.data_paths[seq_idx][sweep_idx]
    points = np.load(self.data_root / filepath)
    coords, feats, labels = (
      points[:, :3],
      points[:, 3:-1],
      points[:, -1],
    )
    feats = np.hstack((feats, coords))
    seq_name, sweep_name = int(filepath[-13:-11]), int(filepath[-10:-4])
    pose = self.poses[seq_name][sweep_name]
    coords = coords @ pose[:3, :3]
    coords += pose[:3, 3]
    return coords, feats, labels
  
  def get_classnames(self):
    classnames = [
      'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person', 'bicyclist', 'motorcyclist', 'road',
      'parking', 'sidewalk', 'other-ground', 'building', 'fence', 'vegetation', 'trunk', 'terrain', 'pole',
      'traffic-sign', 'moving-car', 'moving-bicyclist', 'moving-person', 'moving-motorcyclist',
      'moving-other-vehicle', 'moving-truck',
    ]
    return classnames