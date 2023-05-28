from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch
from . layers import *

__all__ = ['MobileNet', 'mobilenetL', 'mobilenetS']


class MobileNetv3(nn.Module):
    __factory = {
        "L": torchvision.models.mobilenet_v3_large,
        "S": torchvision.models.mobilenet_v3_small,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0, is_export=False, **kwargs):
        super(MobileNetv3, self).__init__()
        #for onnx
        self.my_norm = MyBatchNorm1D()
        self.is_export = is_export

        self.pretrained = pretrained
        self.depth = depth
        self.cut_at_pooling = cut_at_pooling
        # Construct base (pretrained) resnet
        if depth not in MobileNetv3.__factory:
            raise KeyError("Unsupported depth:", depth)
        if depth == "L":
            resnet = MobileNetv3.__factory[depth](weights="IMAGENET1K_V2") #(pretrained=pretrained)
        else:
            resnet = MobileNetv3.__factory[depth](weights="IMAGENET1K_V1")
        
        # if depth >= 50:
        #     # resnet.layer4[0].conv2.stride = (1,1)
        #     # resnet.layer4[0].downsample[0].stride = (1,1)
        self.base = resnet.features
        self.gap = nn.AdaptiveAvgPool2d(1)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = resnet.classifier[0].in_features

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

        resnet = MobileNetv3.__factory[self.depth](pretrained=self.pretrained)
        self.base.load_state_dict(resnet.features.state_dict())


def mobileNetS(**kwargs):
    return MobileNetv3("S", **kwargs)


def mobileNetL(**kwargs):
    return MobileNetv3("L", **kwargs)

