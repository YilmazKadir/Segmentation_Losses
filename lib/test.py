import logging
import os
import warnings
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
from MinkowskiEngine import SparseTensor
from torchmetrics import JaccardIndex

from lib.utils import Timer, AverageMeter, precision_at_one, fast_hist, per_class_iu, \
  get_torch_device, save_predictions


def print_info(iteration,
               max_iteration,
               data_time,
               iter_time,
               has_gt=False,
               losses=None,
               ious=None,
               class_names=None):
  debug_str = "{}/{}: ".format(iteration + 1, max_iteration)
  debug_str += "Data time: {:.4f}, Iter time: {:.4f}".format(data_time, iter_time)

  if has_gt:
    debug_str += "\tLoss {loss.val:.3f} (AVG: {loss.avg:.3f}) \tmIOU {mIOU:.3f}\n".format(
            loss=losses, mIOU=np.nanmean(ious))
    if class_names is not None:
      debug_str += "\nClasses: " + " ".join(class_names) + '\n'
    debug_str += 'IOU: ' + ' '.join('{:.03f}'.format(i) for i in ious) + '\n'

  logging.info(debug_str)


def average_precision(prob_np, target_np):
  num_class = prob_np.shape[1]
  label = label_binarize(target_np, classes=list(range(num_class)))
  with np.errstate(divide='ignore', invalid='ignore'):
    return average_precision_score(label, prob_np, average=None)


def test(model, data_loader, config, has_gt=True):
  device = get_torch_device(config.is_cuda)
  dataset = data_loader.dataset
  num_labels = dataset.NUM_LABELS
  global_timer, data_timer, iter_timer = Timer(), Timer(), Timer()
  criterion = nn.CrossEntropyLoss(ignore_index=config.ignore_mask)
  losses, ious = AverageMeter(), 0
  hist = np.zeros((num_labels, num_labels))

  logging.info('===> Start testing')

  global_timer.tic()
  data_iter = data_loader.__iter__()
  max_iter = len(data_loader)
  max_iter_unique = max_iter

  # Fix batch normalization running mean and std
  model.eval()

  # Clear cache (when run in val mode, cleanup training cache)
  torch.cuda.empty_cache()

  if config.save_prediction:
    save_pred_dir = config.save_pred_dir
    os.makedirs(save_pred_dir, exist_ok=True)
    if os.listdir(save_pred_dir):
      raise ValueError(f'Directory {save_pred_dir} not empty. '
                       'Please remove the existing prediction.')

  with torch.no_grad():
    for iteration in range(max_iter):
      data_timer.tic()
      img, target = data_iter.next()
      
      data_time = data_timer.toc(False)

      # Preprocess input
      iter_timer.tic()

      # Feed forward
      output = model(img.to(device))
      
      pred = output.max(1)[1].int()
      
      iter_time = iter_timer.toc(False)
      
      if has_gt:
        target_np = target.numpy()
        num_sample = target_np.shape[0]
        target = target.to(device).long()
        cross_ent = criterion(output, target)
        losses.update(float(cross_ent), num_sample)
        hist += fast_hist(pred.cpu().numpy().flatten(), target_np.flatten(), num_labels)
        ious = per_class_iu(hist) * 100

      if iteration % config.test_stat_freq == 0 and iteration > 0:
        class_names = dataset.get_classnames()
        print_info(
            iteration,
            max_iter_unique,
            data_time,
            iter_time,
            has_gt,
            losses,
            ious,
            class_names=class_names)

      if iteration % config.empty_cache_freq == 0:
        # Clear cache
        torch.cuda.empty_cache()

  global_time = global_timer.toc(False)

  class_names = dataset.get_classnames()
  print_info(
      iteration,
      max_iter_unique,
      data_time,
      iter_time,
      has_gt,
      losses,
      ious,
      class_names=class_names)
  
  logging.info("Finished test. Elapsed time: {:.4f}".format(global_time))

  return losses.avg, np.nanmean(per_class_iu(hist)) * 100
