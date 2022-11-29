import torch.multiprocessing as mp
try:
  mp.set_start_method('forkserver')  # Reuse process created
except RuntimeError:
  pass

import os
import sys
import json
import logging
from easydict import EasyDict as edict

# Torch packages
import torch

# Train deps
from config import get_config
from lib.test import test
from lib.train import train
from lib.utils import get_torch_device, count_parameters
from lib.dataset import initialize_data_loader
from lib.datasets import load_dataset
from models import load_model

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format=os.uname()[1].split('.')[0] + ' %(asctime)s %(message)s',
    datefmt='%m/%d %H:%M:%S',
    handlers=[ch])


def main():
  config = get_config()
  if config.resume:
    json_config = json.load(open(config.resume + '/config.json', 'r'))
    json_config['resume'] = config.resume
    config = edict(json_config)

  if config.is_cuda and not torch.cuda.is_available():
    raise Exception("No GPU found")
  device = get_torch_device(config.is_cuda)

  logging.info('===> Configurations')
  dconfig = vars(config)
  for k in dconfig:
    logging.info('    {}: {}'.format(k, dconfig[k]))

  DatasetClass = load_dataset(config.dataset)
    
  logging.info('===> Initializing dataloader')
  if config.is_train:
    train_data_loader = initialize_data_loader(
        DatasetClass,
        config,
        phase=config.train_phase,
        num_workers=config.num_workers,
        augment_data=True,
        shuffle=True,
        repeat=True,
        return_inverse=False,
        merge=config.merge,
        batch_size=config.batch_size,
        limit_numpoints=config.train_limit_numpoints)
    val_data_loader = initialize_data_loader(
        DatasetClass,
        config,
        phase=config.val_phase,
        num_workers=config.num_val_workers,
        augment_data=False,
        shuffle=True,
        repeat=False,
        return_inverse=config.test_original_pc,
        merge=False,
        batch_size=config.val_batch_size,
        limit_numpoints=False)
    if train_data_loader.dataset.NUM_IN_CHANNEL is not None:
      num_in_channel = train_data_loader.dataset.NUM_IN_CHANNEL
    else:
      num_in_channel = 3  # RGB color

    num_labels = train_data_loader.dataset.NUM_LABELS
  else:
    test_data_loader = initialize_data_loader(
        DatasetClass,
        config,
        phase=config.test_phase,
        num_workers=config.num_workers,
        augment_data=False,
        shuffle=False,
        repeat=False,
        return_inverse=config.test_original_pc,
        merge=False,
        batch_size=config.test_batch_size,
        limit_numpoints=False)
    if test_data_loader.dataset.NUM_IN_CHANNEL is not None:
      num_in_channel = test_data_loader.dataset.NUM_IN_CHANNEL
    else:
      num_in_channel = 3  # RGB color

    num_labels = test_data_loader.dataset.NUM_LABELS

  logging.info('===> Building model')
  NetClass = load_model(config.model)
  model = NetClass(num_in_channel, num_labels)
  logging.info('===> Number of trainable parameters: {}: {}'.format(NetClass.__name__, count_parameters(model)))
  logging.info(model)
  logging.info('===> Model is on device: {}'.format(device))
  model = model.to(device)

  # Load weights if specified by the parameter.
  if config.weights.lower() != 'none':
    logging.info('===> Loading weights: ' + config.weights)
    state_dict = torch.load(config.weights)['state_dict']
    del state_dict["final.kernel"], state_dict["final.bias"]
    for key in list(state_dict.keys()):
        state_dict[key.replace('model.', '')] = state_dict.pop(key)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print("Missing keys: ", missing_keys)
    print("Unexpected keys: ", unexpected_keys)

  if config.is_train:
    train(model, train_data_loader, val_data_loader, config)
  else:
    test(model, test_data_loader, config)


if __name__ == '__main__':
  __spec__ = None
  main()
