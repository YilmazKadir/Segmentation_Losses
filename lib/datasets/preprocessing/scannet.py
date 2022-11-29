from pathlib import Path
import numpy as np
from lib.pc_utils import read_plyfile
from concurrent.futures import ProcessPoolExecutor
import tqdm

SCANNET_RAW_PATH = Path('/globalwork/data/scannet/scannet')
SCANNET_OUT_PATH = Path('/globalwork/yilmaz/data/processed/minkowski_scannet')
TRAIN_DEST = 'train'
TEST_DEST = 'test'
SUBSETS = {TRAIN_DEST: 'scans', TEST_DEST: 'scans_test'}
POINTCLOUD_FILE = '_vh_clean_2.ply'
BUGS = {
    'train/scene0270_00.npy': 50,
    'train/scene0270_02.npy': 50,
    'train/scene0384_00.npy': 149,
}

def handle_process(path):
  f = Path(path.split(',')[0])
  phase_out_path = Path(path.split(',')[1])
  pointcloud = read_plyfile(f)
  # Make sure alpha value is meaningless.
  assert np.unique(pointcloud[:, -1]).size == 1
  # Load label file.
  label_f = f.parent / (f.stem + '.labels' + f.suffix)
  if label_f.is_file():
    label = read_plyfile(label_f)
    # Sanity check that the pointcloud and its label has same vertices.
    assert pointcloud.shape[0] == label.shape[0]
    assert np.allclose(pointcloud[:, :3], label[:, :3])
  else:  # Label may not exist in test case.
    label = np.zeros_like(pointcloud)
  out_f = phase_out_path / (f.name[:-len(POINTCLOUD_FILE)])
  processed = np.hstack((pointcloud[:, :6], np.array([label[:, -1]]).T))
  np.save(out_f, processed)


path_list = []
for out_path, in_path in tqdm(SUBSETS.items()):
  phase_out_path = SCANNET_OUT_PATH / out_path
  phase_out_path.mkdir(parents=True, exist_ok=True)
  for f in (SCANNET_RAW_PATH / in_path).glob('*/*' + POINTCLOUD_FILE):
    path_list.append(str(f) + ',' + str(phase_out_path))

pool = ProcessPoolExecutor(max_workers=20)
result = list(pool.map(handle_process, path_list))

# Fix bug in the data.
for files, bug_index in BUGS.items():
  print(files)

  for f in SCANNET_OUT_PATH.glob(files):
    pointcloud = np.load(f)
    bug_mask = pointcloud[:, -1] == bug_index
    print(f'Fixing {f} bugged label {bug_index} x {bug_mask.sum()}')
    pointcloud[bug_mask, -1] = 0
    np.save(f, pointcloud)
