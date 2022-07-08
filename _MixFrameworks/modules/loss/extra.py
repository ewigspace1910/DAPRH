import torch
import torch.nn as nn
import torch.nn.functional as F


class AALS(nn.Module):
    """ Agreement-aware label smoothing """
    def __init__(self):
        super(AALS, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

    def forward(self, logits, targets, ca):
        log_preds = self.logsoftmax(logits)  # B * C
        targets = torch.zeros_like(log_preds).scatter_(1, targets.unsqueeze(1), 1)
        uni = (torch.ones_like(log_preds) / log_preds.size(-1)).cuda()

        loss_ce = (- targets * log_preds).sum(1)
        loss_kld = F.kl_div(log_preds, uni, reduction='none').sum(1)
        loss = (ca * loss_ce + (1-ca) * loss_kld).mean()
        return loss


class PGLR(nn.Module):
    """ Part-guided label refinement """
    def __init__(self, lam=0.5):
        super(PGLR, self).__init__()
        self.softmax = nn.Softmax(dim=1).cuda()
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()
        self.lam = lam

    def forward(self, logits_g, logits_p, targets, ca):
        targets = torch.zeros_like(logits_g).scatter_(1, targets.unsqueeze(1), 1)
        w = torch.softmax(ca, dim=1)  # B * P
        w = torch.unsqueeze(w, 1)  # B * 1 * P
        preds_p = self.softmax(logits_p)  # B * numClusters * numParts
        ensembled_preds = (preds_p * w).sum(2).detach()  # B * class_num
        refined_targets = self.lam * targets + (1-self.lam) * ensembled_preds

        log_preds_g = self.logsoftmax(logits_g)
        loss = (-refined_targets * log_preds_g).sum(1).mean()
        return loss
