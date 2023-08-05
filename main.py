import os
import sys
import json
import logging
import torch
from easydict import EasyDict as edict
from config import get_config
from lib.test import test
from lib.train import train
from lib.utils import count_parameters
from torch.utils.data import DataLoader
from lib.datasets import get_dataset_by_name
from models import get_model_by_name

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
    device = torch.device('cuda' if config.is_cuda else 'cpu')

    logging.info('===> Configurations')
    dconfig = vars(config)
    for k in dconfig:
        logging.info('    {}: {}'.format(k, dconfig[k]))

    DatasetClass = get_dataset_by_name(config.dataset)
    
    logging.info('===> Initializing dataloader')
    if config.is_train:
        train_dataset = DatasetClass(config, phase=config.train_phase)
        train_data_loader = DataLoader(dataset=train_dataset,
                                       num_workers=config.num_workers,
                                       batch_size=config.batch_size,
                                       shuffle=True)
        val_dataset = DatasetClass(config, phase=config.val_phase)
        val_data_loader = DataLoader(dataset=val_dataset,
                                     num_workers=config.num_val_workers,
                                     batch_size=config.val_batch_size,
                                     shuffle=False)
        
        num_in_channel = train_data_loader.dataset.NUM_IN_CHANNEL
        num_labels = train_data_loader.dataset.NUM_LABELS
    
    else:
        test_dataset = DatasetClass(config, phase=config.test_phase)
        test_data_loader = DataLoader(dataset=test_dataset,
                                      num_workers=config.num_val_workers,
                                      batch_size=config.test_batch_size,
                                      shuffle=False)
            
        num_in_channel = test_data_loader.dataset.NUM_IN_CHANNEL
        num_labels = test_data_loader.dataset.NUM_LABELS

    logging.info('===> Building model')
    model = get_model_by_name(config.model, in_channels=num_in_channel, out_channels=num_labels)
    logging.info('===> Number of trainable parameters: {}: {}'.format(config.model, count_parameters(model)))
    logging.info(model)
    logging.info('===> Model is on device: {}'.format(device))
    model = model.to(device)

    if config.is_train:
        train(model, train_data_loader, val_data_loader, config)
    else:
        test(model, test_data_loader, config)


if __name__ == '__main__':
    main()
