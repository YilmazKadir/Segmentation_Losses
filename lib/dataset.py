from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class VoxelizationDataset(Dataset):
  IS_TEMPORAL = False
  
  def __init__(self,
               img_paths,
               seg_paths,
               augmentations=None,
               data_root='/',
               ignore_index=255,
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
    self.ignore_index = ignore_index
    self.augment_data = augment_data
    self.config = config

    # map labels not evaluated to ignore_label
    label_map = {}
    n_used = 0
    for l in range(self.NUM_LABELS):
      if l in self.IGNORE_LABELS:
        label_map[l] = self.ignore_index
      else:
        label_map[l] = n_used
        n_used += 1
    label_map[self.ignore_index] = self.ignore_index
    self.label_map = label_map
    self.NUM_LABELS -= len(self.IGNORE_LABELS)

  def __len__(self):
    return len(self.img_paths)
  
  def load_data(self, index):
    pass
  
  def __getitem__(self, index):
    img, seg = self.load_data(index)
    if self.augmentations is not None:
      augmented  = self.augmentations(image=img, mask=seg)
      img, seg  = augmented["image"], augmented["mask"]
    seg = np.vectorize(self.label_map.__getitem__)(seg)
    seg = torch.from_numpy(seg.astype(np.uint8))
    return img, seg


def initialize_data_loader(DatasetClass,
                           config,
                           phase,
                           num_workers,
                           shuffle,
                           augment_data,
                           batch_size):
  dataset = DatasetClass(
    config,
    augment_data=augment_data,
    phase=phase)
  
  data_loader = DataLoader(
    dataset=dataset,
    num_workers=num_workers,
    batch_size=batch_size,
    shuffle=shuffle)

  return data_loader
