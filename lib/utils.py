import json
import logging
import os
import errno
import numpy as np
import torch
from PIL import Image

def checkpoint(model, optimizer, epoch, iteration, config, best_val=None, best_val_iter=None, postfix=None):
    mkdir_p(config.log_dir)
    if postfix is not None:
      filename = f"checkpoint_{config.model}{postfix}.pth"
    else:
      filename = f"checkpoint_{config.model}.pth"
    checkpoint_file = config.log_dir + '/' + filename
    state = {
        'iteration': iteration,
        'epoch': epoch,
        'arch': config.model,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
        }
    if best_val is not None:
      state['best_val'] = best_val
      state['best_val_iter'] = best_val_iter
    json.dump(vars(config), open(config.log_dir + '/config.json', 'w'), indent=4)
    torch.save(state, checkpoint_file)
    logging.info(f"Checkpoint saved to {checkpoint_file}")
    # Delete symlink if it exists
    if os.path.exists(f'{config.log_dir}/weights.pth'):
        os.remove(f'{config.log_dir}/weights.pth')
    # Create symlink
    os.system(f'cd {config.log_dir}; ln -s {filename} weights.pth')


def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    return np.bincount(n * label[k].astype(int) + pred[k], minlength=n**2).reshape(n, n)


def per_class_iu(hist):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):    
            pass
        else:
            raise


def read_txt(path):
    """Read txt file into lines."""
    with open(path) as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]
    return lines


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_info(losses=None,ious=None,class_names=None):        
    debug_str = "\tLoss {losses.val:.3f} (AVG: {losses.avg:.3f}) \tmIOU {mIOU:.3f}\n".format(     
        losses=losses, mIOU=np.nanmean(ious))
    debug_str += "\nClasses: " + " ".join(class_names) + '\n'
    debug_str += 'IOU: ' + ' '.join('{:.03f}'.format(i) for i in ious) + '\n'
    logging.info(debug_str)


def save_image(image_tensor, mask_tensor):
    color_map = [
        (0, 0, 0),         # Class 0: Background - Black
        (255, 0, 0),       # Class 1: Red
        (0, 255, 0),       # Class 2: Green
        (0, 0, 255),       # Class 3: Blue
        (255, 255, 0),     # Class 4: Yellow
        (255, 0, 255),     # Class 5: Magenta
        (0, 255, 255),     # Class 6: Cyan
        (128, 128, 128),   # Class 7: Gray
        (255, 255, 255),   # Class 8: White
    ]
    # Convert the single-channel image tensor to a numpy array and then to 3-channel RGB
    image_array = (image_tensor.squeeze().numpy() * 255).astype(np.uint8)
    image_rgb = np.stack((image_array,) * 3, axis=-1)
    # Convert the mask tensor to a numpy array and apply the color map
    mask_array = mask_tensor.numpy()
    mask_colored = np.zeros((mask_array.shape[0], mask_array.shape[1], 3), dtype=np.uint8)
    for class_idx, color in enumerate(color_map):
        mask_colored[mask_array == class_idx] = color
    # Convert the numpy arrays to PIL images
    image_pil = Image.fromarray(image_rgb)
    mask_pil = Image.fromarray(mask_colored)
    # Save the images
    image_pil.save('image.png')
    mask_pil.save('mask.png')