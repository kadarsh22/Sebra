import torch.nn as nn
import torch

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.sim = nn.CosineSimilarity(dim=1)

    def compute_exp_sim(self, features_anchor, features_):
        """
        Compute sum(sim(anchor, pos)) or sum(sim(anchor, neg))
        """
        sim = self.sim(features_anchor, features_)
        exp_sim = torch.exp(torch.div(sim, self.temperature))
        return exp_sim

    def forward(self, feature_ref, features_pos,  features_neg):
        # Compute negative similarities
        exp_neg = self.compute_exp_sim(feature_ref, features_neg)
        sum_exp_neg = exp_neg.sum(0, keepdim=True)
        exp_pos = self.compute_exp_sim(feature_ref, features_pos)


        log_probs = (torch.log(exp_pos) -
                     torch.log(sum_exp_neg + exp_pos.sum(0, keepdim=True)))
        loss = -1 * log_probs
        del exp_pos;
        del exp_neg;
        del log_probs
        return loss.mean()