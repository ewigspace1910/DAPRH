from __future__ import absolute_import
from this import d

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch

__all__ = [
    "Regnet",
    "regnetY128gf",
    "regnetY32gf",
    "regnetY16gf",
    "regnetY3_2gf",
    "regnetY1_6gf",
    "regnetY400",
    "regnetY800"
]

class Regnet(nn.Module):
    __factory = {
    "regnetY128gf"  : torchvision.models.regnet_y_128gf,
    "regnetY32gf"   : torchvision.models.regnet_y_32gf,
    "regnetY16gf"   : torchvision.models.regnet_y_16gf,
    "regnetY3_2gf"  : torchvision.models.regnet_y_3_2gf,
    "regnetY1_6gf"  : torchvision.models.regnet_y_1_6gf,
    "regnetY400"    : torchvision.models.regnet_y_400mf,
    "regnetY800"    : torchvision.models.regnet_y_800mf
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0, **kwargs):
        super(Regnet, self).__init__()
        self.pretrained = pretrained
        self.depth = depth
        self.cut_at_pooling = cut_at_pooling
        # Construct base (pretrained) model
        if depth not in Regnet.__factory:
            raise KeyError("Unsupported depth:", depth)
        try:
            model = Regnet.__factory[depth](weights="IMAGENET1K_SWAG_LINEAR_V1")
        except:
            model = Regnet.__factory[depth](weights="IMAGENET1K_V2")

        self.base = nn.Sequential(model.stem, model.trunk_output)
        self.gap = nn.AdaptiveAvgPool2d(1)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes
            
            out_planes = model.fc.in_features #check orginal structure if get error!
            
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

        model = Regnet.__factory[self.depth](pretrained=self.pretrained)
        
        model = nn.Sequential(model.stem, model.trunk_output)
        self.base.load_state_dict(model.state_dict()) #check orginal model if get error


####------------------------------####
def regnetY128gf(**kwargs):
    return Regnet("regnetY128gf", **kwargs)

def regnetY32gf(**kwargs):
    return Regnet("regnetY32gf", **kwargs)

def regnetY16gf(**kwargs):
    return Regnet("regnetY16gf", **kwargs)

def regnetY3_2gf(**kwargs):
    return Regnet("regnetY3_2gf", **kwargs)

def regnetY1_6gf(**kwargs):
    return Regnet("regnetY1_6gf", **kwargs)

def regnetY400(**kwargs):
    return Regnet("regnetY400", **kwargs)

def regnetY800(**kwargs):
    return Regnet("regnetY800", **kwargs)




