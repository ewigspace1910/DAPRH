#coding=utf-8
import torch 
import torch.nn as nn 
import torch.nn.parallel
import torch.nn.functional as F 
from torch.nn import Parameter 
import numpy as np 
from ..metric_learning.distance import compute_distance_matrix
from .triplet import _batch_hard

class CenterTripletLoss(nn.Module):
    def __init__(self, margin=0, num_classes=2048, square=False, soft=False):
        super(CenterTripletLoss, self).__init__() 
        self.margin = margin 
        # self.centers = nn.Parameter(torch.randn(num_classes, num_classes)) 
        self.square = square
        self.is_soft=soft
        self.margin = margin
        self.margin_loss = nn.MarginRankingLoss(margin=margin).cuda()
   
    def forward(self, inputs, targets): 
        batch_size = inputs.size(0) 
        #step1: compute center --> BxCxF  | C is number identities and F is feature embds
        # targets_expand = targets.view(batch_size, 1).expand(batch_size, inputs.size(1)) 
        with torch.no_grad():
            centers_batch = self.compute_batch_centers(inputs, targets)
        

        #step2: compute distance sample and all center -> B x C
        mat_dist = compute_distance_matrix(inputs, centers_batch)
        mat_dist = mat_dist if self.square else mat_dist.clamp(min=1e-12).sqrt()  # for numerical stability
    
        # Step3: as normal
        assert mat_dist.size(0) == mat_dist.size(1)
        N = mat_dist.size(0)
        mat_sim = targets.expand(N, N).eq(targets.expand(N, N).t()).float()

        dist_ap, dist_an, _,_ = _batch_hard(mat_dist, mat_sim, indice=True)
        assert dist_an.size(0)==dist_ap.size(0)

        if self.is_soft:
            triple_dist = torch.stack((dist_ap, dist_an), dim=1)
            triple_dist = F.log_softmax(triple_dist, dim=1)
            loss = (- self.margin * triple_dist[:,0] - (1 - self.margin) * triple_dist[:,1]).mean()
            return loss

        y = torch.ones_like(dist_ap)
        loss = self.margin_loss(dist_an, dist_ap, y)

        return loss 



    def compute_batch_centers(self, features, labels):
        # assume `features` is a tensor of shape (batch_size, feature_dim)
        # assume `labels` is a tensor of shape (batch_size,)
        batch_size = features.shape[0]
        num_classes = torch.unique(labels).numel()
        unique_labels = torch.unique(labels)
        centers = torch.zeros((num_classes, features.size(1)), device=features.device)
        # Compute mask for the classes that appear in the batch
        targets_expand = labels.view(batch_size, 1).expand(batch_size, batch_size) #BxB
        mask = labels[:, None] == unique_labels
        # Compute sum of features and counts for each class
        centers = torch.zeros(num_classes, features.size(1), device=features.device)
        # class_counts = torch.zeros(num_classes, device=features.device)
        for i, pid in enumerate(unique_labels): centers[i] = features[mask[:, i]].mean(0)
        
        unique_labels = unique_labels.tolist()
        mask = [unique_labels.index(x) for x in labels.tolist()]
        centers = centers[mask]

        return centers



if __name__ == "__main__":
    L = CenterTripletLoss(num_classes=10)
    x = torch.rand(4, 10)
    print(x)
    y = torch.Tensor([1,20,20,1]).long()
    print("----y---\n", y)
    print(L(x,y))