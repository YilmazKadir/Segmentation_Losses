import torch
import torch.nn as nn
from losses.utils import flatten
from torch.nn.functional import softmax


class TopKLoss(nn.CrossEntropyLoss):
    def __init__(self, ignore_index=255, gamma=0.1):
        super().__init__(reduction='none')
        self.ignore_index = ignore_index
        self.gamma = gamma
      
    def forward(self, input, target):
        input, target = flatten(input, target, self.ignore_index)
        input_prob = torch.gather(softmax(input, dim=1), 1, target.unsqueeze(1))
        values, indices = torch.topk(input_prob, len(target)//10, largest=False)
        cross_entropy = super().forward(input, target)
        losses = torch.pow(1 - input_prob, self.gamma) * cross_entropy
        loss = losses.mean()
        return loss
