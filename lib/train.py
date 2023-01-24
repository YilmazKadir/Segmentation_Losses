import logging
import os.path as osp

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim import AdamW

from lib.test import test
from lib.utils import checkpoint, AverageMeter

from torch.nn import CrossEntropyLoss
from losses.lovasz_softmax import LovaszSoftmaxLoss
from losses.dice import DiceLoss
from losses.jaccard import JaccardLoss
from losses.tversky import TverskyLoss
from losses.focal_tversky import FocalTverskyLoss
from losses.focal import FocalLoss


def validate(model, val_data_loader, writer, curr_iter, config):
  v_loss, v_mIoU = test(model, val_data_loader, config)
  writer.add_scalar('validation/mIoU', v_mIoU, curr_iter)
  writer.add_scalar('validation/loss', v_loss, curr_iter)
  
  return v_mIoU


def train(model, data_loader, val_data_loader, config):
  device = torch.device('cuda' if config.is_cuda else 'cpu')
  model.train()

  # Configuration
  best_val_miou, best_val_iter, curr_iter, epoch = 0, 0, 1, 1
  writer = SummaryWriter(log_dir=config.log_dir)
  losses = AverageMeter()
  
  optimizer = AdamW(model.parameters(), lr=config.lr)
  scheduler = OneCycleLR(optimizer, max_lr=config.lr, total_steps=config.max_epochs*len(data_loader))
  criterion = FocalLoss(ignore_index=config.ignore_index)
  
  logging.info('===> Start training')

  if config.resume:
    checkpoint_fn = config.resume + '/weights.pth'
    if osp.isfile(checkpoint_fn):
      logging.info("=> loading checkpoint '{}'".format(checkpoint_fn))
      
      state = torch.load(checkpoint_fn)
      model.load_state_dict(state['state_dict'])

      curr_iter = state['iteration'] + 1      
      scheduler = OneCycleLR(
        optimizer, max_lr=config.lr, total_steps=config.max_epochs*len(data_loader), last_step=curr_iter)
      optimizer.load_state_dict(state['optimizer'])
      
      if 'best_val' in state:
        best_val_miou = state['best_val']
        best_val_iter = state['best_val_iter']
      logging.info("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_fn, state['epoch']))
    else:
      raise ValueError("=> no checkpoint found at '{}'".format(checkpoint_fn))

  for epoch in range(1, config.max_epochs+1):
    for img, target in data_loader:
      optimizer.zero_grad()
      batch_loss = 0
      
      # Feed forward
      output = model(img.to(device))
      target = target.long().to(device)
      loss = criterion(output, target)
      batch_loss += loss.item()
      loss.backward()
      
      # Update number of steps
      optimizer.step()
      scheduler.step()
      
      losses.update(batch_loss, target.size(0))
      
      if curr_iter % config.stat_freq == 0 or curr_iter == 1:
        lrs = ', '.join(['{:.3e}'.format(x) for x in scheduler.get_last_lr()])
        debug_str = "===> Epoch[{}]({}/{}): Loss {:.4f} LR: {}\t".format(
            epoch, curr_iter, len(data_loader), losses.avg, lrs)
        logging.info(debug_str)
        
        # Write logs
        writer.add_scalar('training/loss', losses.avg, curr_iter)
        writer.add_scalar('training/learning_rate', scheduler.get_last_lr()[0], curr_iter)
        losses.reset()

      # Save current status, save before val to prevent occational mem overflow
      if curr_iter % config.save_freq == 0:
        checkpoint(model, optimizer, epoch, curr_iter, config, best_val_miou, best_val_iter)

      # Validation
      if curr_iter % config.val_freq == 0:
        val_miou = validate(model, val_data_loader, writer, curr_iter, config)
        if val_miou > best_val_miou:
          best_val_miou = val_miou
          best_val_iter = curr_iter
          checkpoint(model, optimizer, epoch, curr_iter, config, best_val_miou, best_val_iter,
                     "best_val")
        logging.info("Current best mIoU: {:.3f} at iter {}".format(best_val_miou, best_val_iter))

        # Recover back
        model.train()

      # End of iteration
      curr_iter += 1

  # Save the final model
  checkpoint(model, optimizer, epoch, curr_iter, config, best_val_miou, best_val_iter)
  val_miou = validate(model, val_data_loader, writer, curr_iter, config)
  if val_miou > best_val_miou:
    best_val_miou = val_miou
    best_val_iter = curr_iter
    checkpoint(model, optimizer, epoch, curr_iter, config, best_val_miou, best_val_iter, "best_val")
  logging.info("Current best mIoU: {:.3f} at iter {}".format(best_val_miou, best_val_iter))
