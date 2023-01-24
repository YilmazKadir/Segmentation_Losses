import torch
import torch.nn as nn
from torch.nn.functional import softmax
from losses.utils import flatten


class DiceLoss(nn.Module):
    def __init__(self, ignore_index=255, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth
      
    def forward(self, input, target):
        input, target = flatten(input, target, self.ignore_index)
        input = softmax(input, dim=1)
        num_classes = input.size(1)
        losses = []
        for c in range(num_classes):
            target_c = (target == c).float()
            input_c = input[:, c]
            
            intersection = (input_c * target_c).sum()
            dice = (2.*intersection + self.smooth)/(input.sum() + target.sum() + self.smooth)
            
            losses.append(1-dice)
        
        losses = torch.stack(losses)
        loss = losses.mean()
        return loss
