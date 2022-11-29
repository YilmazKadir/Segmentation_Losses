import glob
import os
from lib.utils import mkdir_p

SEM_KITTI_OUT_PATH = "/globalwork/yilmaz/data/processed/semantic_kitti"

def generate_splits(sem_kitti_out_path):
  """Takes preprocessed out path and generate txt files"""
  split_path = './splits/sem_kitti'
  mkdir_p(split_path)
  curr_path = os.path.join(sem_kitti_out_path, "test")
  files = glob.glob(os.path.join(curr_path, '*.npy'))
  files = [os.path.relpath(full_path, sem_kitti_out_path) for full_path in files]
  out_txt = os.path.join(split_path, "test.txt")
  with open(out_txt, 'w') as f:
    f.write('\n'.join(files))
      
if __name__ == '__main__':
  generate_splits(SEM_KITTI_OUT_PATH)