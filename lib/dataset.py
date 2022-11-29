from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import MinkowskiEngine as ME

import lib.transforms as t
from lib.dataloader import InfSampler
import cv2
from matplotlib import pyplot as plt
from PIL import Image


class VoxelizationDataset(Dataset):
  IS_TEMPORAL = False
  
  def __init__(self,
               img_paths,
               seg_paths,
               augmentations=None,
               data_root='/',
               ignore_mask=255,
               return_inverse=False,
               augment_data=False,
               config=None):
    
    Dataset.__init__(self)

    # Allows easier path concatenation
    if not isinstance(data_root, Path):
      data_root = Path(data_root)

    self.data_root = data_root
    self.img_paths = img_paths
    self.seg_paths = seg_paths
    self.augmentations = augmentations
    self.ignore_mask = ignore_mask
    self.return_inverse = return_inverse
    self.augment_data = augment_data
    self.config = config

    # map labels not evaluated to ignore_label
    label_map = {}
    n_used = 0
    for l in range(self.NUM_LABELS):
      if l in self.IGNORE_LABELS:
        label_map[l] = self.ignore_mask
      else:
        label_map[l] = n_used
        n_used += 1
    label_map[self.ignore_mask] = self.ignore_mask
    self.label_map = label_map
    self.NUM_LABELS -= len(self.IGNORE_LABELS)

  def __len__(self):
    return len(self.img_paths)
  
  def load_data(self, index):
    img_path = self.data_root / self.img_paths[index]
    seg_path = self.data_root / self.seg_paths[index]
    img = Image.open(img_path).convert('RGB')
    seg = Image.open(seg_path)
    img = img.resize((480,640), resample=Image.BILINEAR)
    seg = seg.resize((480,640), resample=Image.NEAREST)
    return np.array(img), np.array(seg)
  
  def __getitem__(self, index):
    img, seg = self.load_data(index)
    
    if self.augmentations is not None:
      img  = self.augmentations(image=img)["image"]
    
    seg = np.vectorize(self.label_map.__getitem__)(seg)
    seg = torch.from_numpy(seg.astype(np.uint8))
    return img, seg


class TemporalVoxelizationDataset(VoxelizationDataset):

  IS_TEMPORAL = True

  def __init__(self,
               data_paths,
               augmentations=None,
               data_root='/',
               ignore_mask=255,
               return_inverse=False,
               augment_data=False,
               config=None,
               merge=False,
               temporal_dilation=1,
               temporal_numseq=3):
    VoxelizationDataset.__init__(
      self,
      data_paths,
      augmentations=augmentations,
      data_root=data_root,
      ignore_mask=ignore_mask,
      return_inverse=return_inverse,
      augment_data=augment_data,
      config=config,
      merge=merge)
    self.temporal_dilation = temporal_dilation
    self.temporal_numseq = temporal_numseq
    temporal_window = temporal_dilation * (temporal_numseq - 1) + 1
    self.numels = [len(p) - temporal_window + 1 for p in self.data_paths]
    if any([numel <= 0 for numel in self.numels]):
      raise ValueError('Your temporal window configuration is too wide for '
                       'this dataset. Please change the configuration.')
  
  def __len__(self):
    return sum(self.numels)
  
  def __getitem__(self, index):
    for seq_idx, numel in enumerate(self.numels):
      if index >= numel:
        index -= numel
      else:
        break

    numseq = self.temporal_numseq
    if self.augment_data and self.config.temporal_rand_numseq:
      numseq = random.randrange(1, self.temporal_numseq + 1)
    dilations = [self.temporal_dilation for i in range(numseq - 1)]
    if self.augment_data and self.config.temporal_rand_dilation:
      dilations = [random.randrange(1, self.temporal_dilation + 1) for i in range(numseq - 1)]
    
    points = [self.load_world_pointcloud(seq_idx, index + sum(dilations[:i])) for i in range(numseq)]
    coords_t, feats_t, labels_t = zip(*points)
    
    if self.augmentations is not None:
      coords_t, feats_t, labels_t = self.augmentations(coords_t, feats_t, labels_t)
    
    coords_list, feats_list, labels_list, inverse_list = [], [], [], []
    for coords, feats, labels in zip(coords_t, feats_t, labels_t):
      coords, feats, labels, _, inverse_map = ME.utils.sparse_quantize(
        coords, feats, labels, self.ignore_mask, True, True, quantization_size=self.VOXEL_SIZE)
      
      coords_list.append(coords)
      feats_list.append(feats)
      labels_list.append(labels)
      inverse_list.append(inverse_map)
        
    coords = np.vstack([
      np.hstack((C, np.ones((C.shape[0], 1)) * i)) for i, C in enumerate(coords_list)
    ])
    feats = np.vstack(feats_list)
    labels = np.hstack(labels_list)
    
    if self.IGNORE_LABELS is not None:
      labels = np.array([self.label_map[x] for x in labels], dtype=np.int)
        
    if not self.return_inverse:
      return coords, feats, labels
    else:
      inverse_maps = np.vstack([
        np.hstack((inverse[:, np.newaxis], np.ones((inverse.shape[0], 1)) * i)) for i, inverse in enumerate(inverse_list)
      ])
      original_labels = np.hstack(labels_t)
      if self.IGNORE_LABELS is not None:
        original_labels = np.array([self.label_map[x] for x in original_labels], dtype=np.int)
      return coords, feats, labels, original_labels, inverse_maps

    
def initialize_data_loader(DatasetClass,
                           config,
                           phase,
                           num_workers,
                           shuffle,
                           repeat,
                           augment_data,
                           batch_size,
                           limit_numpoints,
                           return_inverse,
                           merge):
  collate_fn = t.cfl_collate_fn_factory(limit_numpoints, return_inverse)
  dataset = DatasetClass(
      config,
      augment_data=augment_data,
      return_inverse=return_inverse,
      phase=phase)
  data_args = {
      'dataset': dataset,
      'num_workers': num_workers,
      'batch_size': batch_size,
  }
  if repeat:
    data_args['sampler'] = InfSampler(dataset, shuffle)
  else:
    data_args['shuffle'] = shuffle

  data_loader = DataLoader(**data_args)

  return data_loader
