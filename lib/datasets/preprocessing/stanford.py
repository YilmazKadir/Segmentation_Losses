import numpy as np
import os
from pathlib import Path

STANFORD_3D_IN_PATH = '/work/yilmaz/data/raw/s3dis/'
STANFORD_3D_OUT_PATH = '/globalwork/yilmaz/data/processed/minkowski_S3DIS/'


class Stanford3DDatasetConverter:

  CLASSES = [
    'clutter', 'beam', 'board', 'bookcase', 'ceiling', 'chair', 'column', 'door', 'floor', 'sofa',
    'table', 'wall', 'window'
  ]
  TRAIN_TEXT = 'train'
  VAL_TEXT = 'val'
  TEST_TEXT = 'test'

  @classmethod
  def convert_to_npy(cls, root_path, out_path):
    (Path(out_path) / "train").mkdir(parents=True, exist_ok=True)
    (Path(out_path) / "validation").mkdir(parents=True, exist_ok=True)
    
    for i in range(1, 7):
      area_dir = root_path + f"/Area_{i}"
      for room_dir in (os.listdir(area_dir)):
        if room_dir[0] == ".":
          continue
        file_dir = area_dir + "/" + room_dir + "/Annotations"
        scene_array = np.zeros((0,7), dtype=float)
        for instance_dir in os.listdir(file_dir):
          if instance_dir[-3:] != "txt":
            continue
          instance_array = np.loadtxt(file_dir + "/" + instance_dir)

          if instance_dir[0:4] == "beam":
            label_array = np.ones((len(instance_array),1), dtype=float) * 1
          elif instance_dir[0:4] == "door":
            label_array = np.ones((len(instance_array),1), dtype=float) * 7
          elif instance_dir[0:4] == "wall":
            label_array = np.ones((len(instance_array),1), dtype=float) * 11
          elif instance_dir[0:4] == "sofa":
            label_array = np.ones((len(instance_array),1), dtype=float) * 9
          elif instance_dir[0:5] == "chair":
            label_array = np.ones((len(instance_array),1), dtype=float) * 5
          elif instance_dir[0:5] == "floor":
            label_array = np.ones((len(instance_array),1), dtype=float) * 8
          elif instance_dir[0:5] == "table":
            label_array = np.ones((len(instance_array),1), dtype=float) * 10
          elif instance_dir[0:5] == "board":
            label_array = np.ones((len(instance_array),1), dtype=float) * 2
          elif instance_dir[0:6] == "column":
            label_array = np.ones((len(instance_array),1), dtype=float) * 6
          elif instance_dir[0:6] == "window":
            label_array = np.ones((len(instance_array),1), dtype=float) * 12
          elif instance_dir[0:6] == "stairs":
            label_array = np.ones((len(instance_array),1), dtype=float) * 0
          elif instance_dir[0:7] == "ceiling":
            label_array = np.ones((len(instance_array),1), dtype=float) * 4
          elif instance_dir[0:7] == "clutter":
            label_array = np.ones((len(instance_array),1), dtype=float) * 0
          elif instance_dir[0:8] == "bookcase":
            label_array = np.ones((len(instance_array),1), dtype=float) * 3
          else:
            print("Incorrect label")
          instance_array = np.hstack((instance_array, label_array))
          scene_array = np.vstack((scene_array, instance_array))
        if i != 5:
          np.save(out_path + "train" + f"/Area_{i}_{room_dir}", scene_array)
        else:
          np.save(out_path + "validation" + f"/Area_{i}_{room_dir}", scene_array)


if __name__ == '__main__':
  Stanford3DDatasetConverter.convert_to_npy(STANFORD_3D_IN_PATH, STANFORD_3D_OUT_PATH)