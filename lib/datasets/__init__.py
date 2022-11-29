import lib.datasets.stanford as stanford
import lib.datasets.scannet as scannet
import lib.datasets.kitti as kitti
import lib.datasets.ade20k as ade20k
DATASETS = []


def add_datasets(module):
  DATASETS.extend([getattr(module, a) for a in dir(module) if 'Dataset' in a])


add_datasets(stanford)
add_datasets(scannet)
add_datasets(kitti)
add_datasets(ade20k)

def load_dataset(name):
  '''Creates and returns an instance of the datasets given its name.
  '''
  # Find the model class from its name
  mdict = {dataset.__name__: dataset for dataset in DATASETS}
  if name not in mdict:
    print('Invalid dataset index. Options are:')
    # Display a list of valid dataset names
    for dataset in DATASETS:
      print('\t* {}'.format(dataset.__name__))
    raise ValueError(f'Dataset {name} not defined')
  DatasetClass = mdict[name]

  return DatasetClass
