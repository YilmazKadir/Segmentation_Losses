import torch
import torch.nn as nn
from losses.utils import flatten
from torch.nn.functional import softmax


class FocalLoss(nn.CrossEntropyLoss):
    def __init__(self, ignore_index=255, gamma=0.1):
        super().__init__(ignore_index=ignore_index, reduction='none')
        self.ignore_index = ignore_index
        self.gamma = gamma
      
    def forward(self, input, target):
        cross_entropy = super().forward(input, target)
        target = target * (target != self.ignore_index).long()
        input_prob = torch.gather(softmax(input, dim=1), 1, target.unsqueeze(1))
        losses = torch.pow(1 - input_prob, self.gamma) * cross_entropy
        loss = losses.mean()
        return loss
