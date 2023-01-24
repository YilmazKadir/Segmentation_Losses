import torch
import torch.nn as nn
from losses.utils import flatten
from torch.nn.functional import softmax


class TverskyLoss(nn.Module):
    def __init__(self, ignore_index=255, smooth=1.0, alpha=0.5, beta=0.5):
        super(TverskyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta

    def forward(self, input, target):
        input, target = flatten(input, target, self.ignore_index)
        input = softmax(input, dim=1)
        num_classes = input.size(1)
        losses = []
        for c in range(num_classes):
            target_c = (target == c).float()
            input_c = input[:, c]
            
            t_p = (input_c * target_c).sum()
            f_p = ((1-target_c) * input_c).sum()
            f_n = (target_c * (1-input_c)).sum()
            tversky = (t_p + self.smooth) / (t_p + self.alpha*f_p + self.beta*f_n + self.smooth)
            
            losses.append(1-tversky)
        
        losses = torch.stack(losses)
        loss = losses.mean()
        return loss