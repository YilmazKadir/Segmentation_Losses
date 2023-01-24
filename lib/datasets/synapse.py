import logging
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

from lib.dataset import VoxelizationDataset
from lib.utils import read_txt


class SynapseDataset(VoxelizationDataset):
  NUM_IN_CHANNEL = 1
  NUM_LABELS = 9
  IGNORE_LABELS = ()

  def __init__(self,
               config,
               augment_data=True,
               return_inverse=False,
               phase="train"):
    data_root = config.synapse_path
    img_paths = read_txt("splits/synapse/" + phase + ".txt")
    logging.info('Loading {}: {}'.format(self.__class__.__name__, phase))

    if augment_data:
      augmentations = A.Compose([
        A.RandomRotate90(),
        A.Rotate(limit=20),
        A.HorizontalFlip(),
        ToTensorV2() ,
      ])
    else:
      augmentations = A.Compose([
        ToTensorV2(),
      ])
    
    super().__init__(
      img_paths,
      None,
      data_root=data_root,
      augmentations=augmentations,
      ignore_mask=config.ignore_mask,
      return_inverse=return_inverse,
      augment_data=augment_data,
      config=config)
  
  def load_data(self, index):
    filepath = self.img_paths[index]
    data = np.load(self.data_root / filepath)
    img, seg = data['image'], data['label']
    # img = Image.open(img_path).convert('RGB')
    # seg = Image.open(seg_path)
    # img = img.resize((480,640), resample=Image.BILINEAR)
    # seg = seg.resize((480,640), resample=Image.NEAREST)
    return img, seg
  
  def get_classnames(self):
    classnames = [
      'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
      'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator',
      'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']
    return classnames