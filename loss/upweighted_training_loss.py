import torch
import torch.nn as nn
import torch.nn.functional as F


class UpweightedTrainingLoss(nn.Module):

    def __init__(self, beta_inverse=0.7):
        super(UpweightedTrainingLoss, self).__init__()
        self.beta_inverse = beta_inverse

    def forward(self, logits, targets):
        p = F.softmax(logits, dim=1)
        py = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        loss_weight = (py.squeeze().detach() ** self.beta_inverse)
        loss = F.cross_entropy(logits, targets, reduction='none') * (loss_weight)
        return loss
