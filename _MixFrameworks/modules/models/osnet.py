from __future__ import absolute_import

from  .part import orginal

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch
from . layers import *

__all__ = [  "OSNet", "osnet0_25", "osnet0_5",   "osnet0_75",   "osnet1_0",   "osnet1_0ibt" ]

class OSNet(nn.Module):
    __factory = {
        "1_0": orginal.osnet_x1_0,
        "1_0ibn": orginal.osnet_ibn_x1_0,
        "0_75": orginal.osnet_x0_75,
        "0_5": orginal.osnet_x0_5,
        "0_25": orginal.osnet_x0_25       
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0, is_export=False, **kwargs):
        super(OSNet, self).__init__()
        #for onnx
        self.my_norm = MyBatchNorm1D()
        self.is_export = is_export

        self.pretrained = pretrained
        self.depth = depth
        self.cut_at_pooling = cut_at_pooling
        # Construct base (pretrained) model
        if depth not in OSNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        model = OSNet.__factory[depth]()

        self.base = nn.Sequential(model.conv1, model.maxpool, model.conv2, model.conv3,
                            model.conv4, model.conv5) #ignore last relu layer 
        self.gap = nn.AdaptiveAvgPool2d(1)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes
            
            
            out_planes = model.classifier.in_features #check orginal structure if get error!
            
            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal_(self.feat.weight, mode='fan_out')
                init.constant_(self.feat.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
                self.feat_bn = nn.BatchNorm1d(self.num_features)
            self.feat_bn.bias.requires_grad_(False)
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes, bias=False)
                init.normal_(self.classifier.weight, std=0.001)
        init.constant_(self.feat_bn.weight, 1)
        init.constant_(self.feat_bn.bias, 0)

        if not pretrained:
            self.reset_params()

    def forward(self, x, feature_withbn=False):
        x = self.base(x) # [bs, channel, 16, 8]

        x = self.gap(x)
        #for onnx
        if self.is_export:
            x = x.squeeze(3).squeeze(2)
            if self.has_embedding:
                x = self.feat(x)
            out = self.my_norm(x, self.feat_bn)
            return out
        x = x.view(x.size(0), -1)
        if self.cut_at_pooling:
            return x

        if self.has_embedding:
            bn_x = self.feat_bn(self.feat(x))
        else:
            bn_x = self.feat_bn(x)

        if self.training is False:
            bn_x = F.normalize(bn_x)
            return bn_x

        if self.norm:
            bn_x = F.normalize(bn_x)
        elif self.has_embedding:
            bn_x = F.relu(bn_x)

        if self.dropout > 0:
            bn_x = self.drop(bn_x)

        if self.num_classes > 0:
            prob = self.classifier(bn_x)
        else:
            return x, bn_x

        if feature_withbn:
            return bn_x, prob
        return x, prob

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

        model = OSNet.__factory[self.depth](pretrained=self.pretrained)
        
        self.base[0].load_state_dict(model.conv1.state_dict())
        self.base[1].load_state_dict(model.maxpool.state_dict())
        self.base[2].load_state_dict(model.conv2.state_dict())
        self.base[3].load_state_dict(model.conv3.state_dict())
        self.base[4].load_state_dict(model.conv4.state_dict())
        self.base[5].load_state_dict(model.conv5.state_dict())



####------------------------------####
def osnet0_25(**kwargs):
    return OSNet("0_25", **kwargs)

def osnet0_5(**kwargs):
    return OSNet("0_5", **kwargs)

def osnet0_75(**kwargs):
    return OSNet("0_75", **kwargs)

def osnet1_0(**kwargs):
    return OSNet("1_0", **kwargs)

def osnet1_0ibt(**kwargs):
    return OSNet("1_0ibt", **kwargs)