import numpy as np
import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from lib.utils import read_txt
from pathlib import Path
from PIL import Image


class CityScapesDataset(Dataset):
    NUM_IN_CHANNEL = 3
    NUM_LABELS = 34
    IGNORE_LABELS = (0,1,2,3,4,5,6,9,10,14,15,16,18,29,30)

    def __init__(self, config, phase="train"):
        self.ignore_index = config.ignore_index
        self.data_root = Path(config.cityscapes_path)
        self.img_paths = read_txt("splits/cityscapes/" + phase + ".txt")
        self.seg_paths=[]
        for path in self.img_paths:
            path = path.replace('leftImg8bit/', 'gtFine/')
            path = path.replace('_leftImg8bit', '_gtFine_labelIds')
            self.seg_paths.append(path)

        if phase == config.train_phase:
            self.augmentations = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RGBShift(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(),
                ToTensorV2(),
                ])
        else:
            self.augmentations = A.Compose([
                A.Normalize(),
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
        img_path = self.data_root / self.img_paths[index]
        seg_path = self.data_root / self.seg_paths[index]
        img = Image.open(img_path).convert('RGB')
        seg = Image.open(seg_path)
        img = img.resize((512,512), resample=Image.BILINEAR)
        seg = seg.resize((512,512), resample=Image.NEAREST)
        if self.augmentations is not None:
            augmented  = self.augmentations(image=np.array(img), mask=np.array(seg))
            img, seg  = augmented["image"], augmented["mask"]
        seg = np.vectorize(self.label_map.__getitem__)(seg)
        seg = torch.from_numpy(seg.astype(np.uint8))
        return img, seg

    def get_classnames(self):
        return ["Road", "Sidewalk", "Building", "Wall", "Fence", "Pole", "Traffic-Light", "Traffic-Sign",
                "Vegetation", "Terrain", "Sky", "Person", "Rider", "Car", "Truck", "Bus", "Train",
                "Motorcycle", "Bicycle"]