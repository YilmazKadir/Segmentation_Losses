import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from lib.utils import read_txt
from pathlib import Path


class SynapseDataset(Dataset):
    NUM_IN_CHANNEL = 1
    NUM_LABELS = 9
    IGNORE_LABELS = (0,)

    def __init__(self, config, phase="train"):
        self.ignore_index = config.ignore_index
        self.data_root = Path(config.synapse_path)
        self.img_paths = read_txt("splits/synapse/" + phase + ".txt")
        if phase == config.train_phase:
            self.augmentations = A.Compose([
                A.RandomRotate90(),
                A.Rotate(limit=20),
                A.HorizontalFlip(),
                ToTensorV2(),
                ])
        else:
            self.augmentations = A.Compose([
                ToTensorV2(),
                ])
        
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

    def __getitem__(self, index):
        filepath = self.img_paths[index]
        data = np.load(self.data_root / filepath)
        img, seg = data['image'], data['label']
        seg = np.vectorize(self.label_map.__getitem__)(seg)
        if np.all(seg == self.ignore_index):
            return self.__getitem__(np.random.randint(self.__len__()))
        if self.augmentations is not None:
            augmented  = self.augmentations(image=img, mask=seg)
            img, seg  = augmented["image"], augmented["mask"]
        return img, seg

    def get_classnames(self):
        return ["Aorta", "Gallbladder", "Left-Kidney", "Right-Kidney", "Liver", "Pancreas",
                "Spleen", "Stomach"]