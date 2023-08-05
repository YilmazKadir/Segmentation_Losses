import logging
import numpy as np
import torch
import torch.nn as nn
from torchmetrics import JaccardIndex
from lib.utils import AverageMeter, fast_hist, per_class_iu, print_info


def test(model, data_loader, config):
    device = torch.device('cuda' if config.is_cuda else 'cpu')
    dataset = data_loader.dataset
    num_labels = dataset.NUM_LABELS
    criterion = nn.CrossEntropyLoss(ignore_index=config.ignore_index)
    losses, ious = AverageMeter(), 0
    hist = np.zeros((num_labels, num_labels))

    logging.info('===> Start testing')
    model.eval()

    # Clear cache (when run in val mode, cleanup training cache)
    torch.cuda.empty_cache()

    with torch.no_grad():
        for img, target in data_loader:
            # Feed forward
            output = model(img.to(device))
            pred = output.max(1)[1].int()
            target_np = target.numpy()
            num_sample = target_np.shape[0]
            target = target.to(device).long()
            loss = criterion(output, target)
            losses.update(float(loss), num_sample)
            hist += fast_hist(pred.cpu().numpy().flatten(), target_np.flatten(), num_labels)
            ious = per_class_iu(hist) * 100

    print_info(losses, ious, class_names=dataset.get_classnames())

    logging.info("Finished test")

    return losses.avg, np.nanmean(per_class_iu(hist)) * 100
