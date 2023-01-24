import logging
import numpy as np
import torch
import torch.nn as nn
from torchmetrics import JaccardIndex

from lib.utils import AverageMeter, fast_hist, per_class_iu


def print_info(has_gt=True,
               losses=None,
               ious=None,
               class_names=None):
  if has_gt:
    debug_str = "\tLoss {loss.val:.3f} (AVG: {loss.avg:.3f}) \tmIOU {mIOU:.3f}\n".format(
            loss=losses, mIOU=np.nanmean(ious))
    # if class_names is not None:
    #   debug_str += "\nClasses: " + " ".join(class_names) + '\n'
    # debug_str += 'IOU: ' + ' '.join('{:.03f}'.format(i) for i in ious) + '\n'

  logging.info(debug_str)


def test(model, data_loader, config, has_gt=True):
  device = torch.device('cuda' if config.is_cuda else 'cpu')
  dataset = data_loader.dataset
  num_labels = dataset.NUM_LABELS
  criterion = nn.CrossEntropyLoss(ignore_index=config.ignore_index)
  losses, ious = AverageMeter(), 0
  hist = np.zeros((num_labels, num_labels))
  
  logging.info('===> Start testing')
  
  # Fix batch normalization running mean and std
  model.eval()

  # Clear cache (when run in val mode, cleanup training cache)
  torch.cuda.empty_cache()
  
  with torch.no_grad():
    for img, target in data_loader:
      # Feed forward
      output = model(img.to(device))
      
      pred = output.max(1)[1].int()
            
      if has_gt:
        target_np = target.numpy()
        num_sample = target_np.shape[0]
        target = target.to(device).long()
        cross_ent = criterion(output, target)
        losses.update(float(cross_ent), num_sample)
        hist += fast_hist(pred.cpu().numpy().flatten(), target_np.flatten(), num_labels)
        ious = per_class_iu(hist) * 100


  class_names = dataset.get_classnames()
  print_info(
      has_gt,
      losses,
      ious,
      class_names=class_names)
  
  logging.info("Finished test")

  return losses.avg, np.nanmean(per_class_iu(hist)) * 100
