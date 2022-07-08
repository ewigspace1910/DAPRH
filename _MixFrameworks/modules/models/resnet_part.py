from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch


__all__ = ['ResNetPart', 'resnet18part', 'resnet34part', 'resnet50part', 'resnet101part',
           'resnet152part']

class ResNetPart(nn.Module):
    """ ResNet with part features by uniform partitioning """
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                num_parts=3, num_classes=0, num_features=0, norm=False, dropout=0):
        super(ResNetPart, self).__init__()
        self.pretrained = pretrained
        self.depth = depth
        self.cut_at_pooling = cut_at_pooling
        # Construct base (pretrained) resnet
        if depth not in ResNetPart.__factory:
            raise KeyError("Unsupported depth:", depth)
        resnet = ResNetPart.__factory[depth](pretrained=pretrained)
        resnet.layer4[0].conv2.stride = (1,1)
        resnet.layer4[0].downsample[0].stride = (1,1)

        self.base = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        self.gap = nn.AdaptiveAvgPool2d(1)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = resnet.fc.in_features

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

            #for classify
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes, bias=False)
                init.normal_(self.classifier.weight, std=0.001)
        init.constant_(self.feat_bn.weight, 1)
        init.constant_(self.feat_bn.bias, 0)

        ##########################
        #ADD ideal in PPLR 
        #https://github.com/ewigspace1910/Paper-Notes-Deep-Learning/blob/main/Computer%20Vision/3.Person%20ReID/PPLR.md
        self.num_parts = num_parts
        self.rap = nn.AdaptiveAvgPool2d((self.num_parts, 1))

        # # global feature classifiers
        # self.bnneck = nn.BatchNorm1d(2048)
        # init.constant_(self.bnneck.weight, 1)
        # init.constant_(self.bnneck.bias, 0)
        # self.bnneck.bias.requires_grad_(False)

        # self.classifier = nn.Linear(2048, self.num_classes, bias=False)
        # init.normal_(self.classifier.weight, std=0.001)

        # part feature classifiers
        for i in range(self.num_parts):
            name = 'bnneck' + str(i)
            setattr(self, name, nn.BatchNorm1d(2048))
            init.constant_(getattr(self, name).weight, 1)
            init.constant_(getattr(self, name).bias, 0)
            getattr(self, name).bias.requires_grad_(False)

            name = 'classifier' + str(i)
            setattr(self, name, nn.Linear(2048, self.num_classes, bias=False))

        if not pretrained:
            self.reset_params()

    def forward(self, x): #just training and cant using to evaluate
        x = self.base(x) # [bs, 2048, 16, 8]

        f_g = self.gap(x) # [bs, 2048, 1, 1]
        f_g = f_g.view(x.size(0), -1)

        if self.cut_at_pooling:
            return f_g

        if self.has_embedding:
            bn_x = self.feat_bn(self.feat(f_g))
        else:
            bn_x = self.feat_bn(f_g) #[bs, 2048] #global feature

        if self.training is False:
            bn_x = F.normalize(bn_x)
            return bn_x  #when extract features
        
        ######################
        if self.norm:
            bn_x = F.normalize(bn_x)
        elif self.has_embedding:
            bn_x = F.relu(bn_x) 
        if self.dropout > 0:
            bn_x_ = self.drop(bn_x)
        logits_g = self.classifier(bn_x_)

        f_p = self.rap(x)
        f_p = f_p.view(f_p.size(0), f_p.size(1), -1)

        logits_p = []
        fs_p = []
        for i in range(self.num_parts):
            f_p_i = f_p[:, :, i]
            f_p_i = getattr(self, 'bnneck' + str(i))(f_p_i)
            logits_p_i = getattr(self, 'classifier' + str(i))(f_p_i)
            logits_p.append(logits_p_i)
            fs_p.append(f_p_i)

        fs_p = torch.stack(fs_p, dim=-1)
        logits_p = torch.stack(logits_p, dim=-1)

        return bn_x, fs_p, logits_g, logits_p #feature global, feature parts [1,2,3]

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

    def extract_all_features(self, x):
        x = self.base(x)

        f_g = self.gap(x)
        f_g = f_g.view(x.size(0), -1)
        if self.has_embedding:
            bn_x = self.feat_bn(self.feat(f_g))
        else:
            bn_x = self.feat_bn(f_g) #[bs, 2048]
        f_g = F.normalize(bn_x)

        f_p = self.rap(x)
        f_p = f_p.view(f_p.size(0), f_p.size(1), -1)

        fs_p = []
        for i in range(self.num_parts):
            f_p_i = f_p[:, :, i]
            f_p_i = getattr(self, 'bnneck' + str(i))(f_p_i)
            f_p_i = F.normalize(f_p_i)
            fs_p.append(f_p_i)
        fs_p = torch.stack(fs_p, dim=-1)

        return f_g, fs_p


def resnet18part(**kwargs):
    return ResNetPart(18, **kwargs)


def resnet34part(**kwargs):
    return ResNetPart(34, **kwargs)


def resnet50part(**kwargs):
    return ResNetPart(50, **kwargs)


def resnet101part(**kwargs):
    return ResNetPart(101, **kwargs)


def resnet152part(**kwargs):
    return ResNetPart(152, **kwargs)