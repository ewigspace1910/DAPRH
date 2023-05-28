"""
Yoonki Cho, Woo Jae Kim, Seunghoon Hong, Sung-Eui Yoon. 28 Mar 2022
url:https://arxiv.org/abs/2203.14675
github:https://github.com/yoonkicho/pplr
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


"""
combine  neighbour centroid score to refine pseudo label.
"""
class UET(nn.Module):
    def __init__(self, alpha=0.3):
        super(UET, self).__init__()
        self.softmax = nn.Softmax(dim=1).cuda()
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()
        self.alpha = alpha


    def forward(self, logits, targets, aff_score=None, alphas=None, **kwargs):
        #logits : BxC    C: num_classes
        #neighbour : NxC   N: num_samples
        #target  : Bx1
        #aff_score: BxN
        #ca_score : BxP    P:num_parts
        targets = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1)
        
        if alphas is None:
            refined_targets = (1-self.alpha) * targets + self.alpha * aff_score
        else:
            refined_targets = (1-alphas) * targets + alphas * aff_score
        log_preds = self.logsoftmax(logits)
        loss = (-refined_targets * log_preds).sum(1)
        loss = loss.mean()
        return loss
