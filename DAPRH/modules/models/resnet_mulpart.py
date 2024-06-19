from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch

from .layer import Bottleneck, conv1x1, conv3x3, MLP, CBAM


__all__ = ['ResNetMulpart', 'resnet18mulpart', 'resnet34mulpart', 'resnet50mulpart', 'resnet101mulpart',
           'resnet152mulpart']


class ResNetMulpart(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, num_features=0, num_part=3, num_classes=0, **kwargs):
        super(ResNetMulpart, self).__init__()

        self.pretrained = pretrained
        self.depth = depth
        self.num_parts=num_part
        # Construct base (pretrained) resnet
        if depth not in ResNetMulpart.__factory: raise KeyError("Unsupported depth:", depth)
        if depth >= 50: resnet = ResNetMulpart.__factory[depth](pretrained="IMAGENET1K_V2") #(pretrained=pretrained)
        else: resnet = ResNetMulpart.__factory[depth](pretrained=pretrained)
        if depth >= 50:
            resnet.layer4[0].conv2.stride = (1,1)
            resnet.layer4[0].downsample[0].stride = (1,1)

        self.base = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)


        self.num_features = num_features
        self.has_embedding = num_features > 0
        self.num_classes = num_classes
        out_planes = resnet.fc.in_features

        # Append new Last FC layers
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
        if self.num_classes > 0:
            self.classifier = nn.Linear(self.num_features, self.num_classes, bias=False)
            init.normal_(self.classifier.weight, std=0.001)
        init.constant_(self.feat_bn.weight, 1)
        init.constant_(self.feat_bn.bias, 0)

        self.feat_bn_concat = nn.BatchNorm1d(self.num_features * self.num_parts)
        self.feat_bn_concat.bias.requires_grad_(False)
        init.constant_(self.feat_bn_concat.weight, 1)
        init.constant_(self.feat_bn_concat.bias, 0)            
        if self.num_classes > 0:
            self.classifier_concat = nn.Linear(self.num_features * self.num_parts, self.num_classes, bias=False)
            init.normal_(self.classifier_concat.weight, std=0.001)      

        # sub branches
        self.hrap = nn.AdaptiveAvgPool2d((self.num_parts, 1)) 
        self.hrmp = nn.AdaptiveMaxPool2d((self.num_parts, 1)) 

        if not pretrained: self.reset_params()

    def forward(self, x, finetune=False, joint=False):
        '''Denotes: B-batchsize, C-number classes, K-number parts, D-dimention embd
        
        '''
        featuremap = self.base(x) # [bs, 2048, 16, 8]
        xa = self.gap(featuremap)
        xm = self.gmp(featuremap)
        
        xa = xa.view(xa.size(0), -1)
        xm = xm.view(xm.size(0), -1)
        
        if self.has_embedding: 
            bn_xa = self.feat_bn(self.feat(xm))
            bn_xm = self.feat_bn(self.feat(xa))
        else: 
            bn_xm   = self.feat_bn(xm)
            bn_xa   = self.feat_bn(xa)

        if self.training is False and finetune is False and joint is False:
            bn_xm = F.normalize(bn_xm)
            return bn_xm


        
        fm_p = self.hrmp(featuremap)
        fm_p = fm_p.view(fm_p.size(0), fm_p.size(1), -1) #Bx2048x1x1xK-->Bx2048xK
        fm_p_concat = fm_p.transpose(1,2).flatten(1)   #Bx2048xK-->BxKx2048->Bx(2048*K)

        f_p = fm_p_concat #+ fa_p_concat

        bn_p = self.feat_bn_concat(f_p)
        if self.training is False and finetune is False and joint:
            bn_xm = F.normalize(bn_xm)
            bn_fx = F.normalize(bn_p)
            return bn_xm, bn_fx

        logits_p = self.classifier_concat(bn_p)
        logits_g = self.classifier(bn_xa)

        if finetune:
            nfs_x = F.normalize(bn_xm)
            nfs_hp = F.normalize(bn_p)

            return [bn_xm, bn_p], [nfs_x, nfs_hp],[logits_g, logits_p] 
        else: return [bn_xm, bn_p], [logits_g, logits_p]




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

        resnet = ResNetMulpart.__factory[self.depth](pretrained=self.pretrained)
        self.base[0].load_state_dict(resnet.conv1.state_dict())
        self.base[1].load_state_dict(resnet.bn1.state_dict())
        self.base[2].load_state_dict(resnet.relu.state_dict())
        self.base[3].load_state_dict(resnet.maxpool.state_dict())
        self.base[4].load_state_dict(resnet.layer1.state_dict())
        self.base[5].load_state_dict(resnet.layer2.state_dict())
        self.base[6].load_state_dict(resnet.layer3.state_dict())
        self.base[7].load_state_dict(resnet.layer4.state_dict())




def resnet18mulpart(**kwargs):
    return ResNetMulpart(18, **kwargs)


def resnet34mulpart(**kwargs):
    return ResNetMulpart(34, **kwargs)


def resnet50mulpart(**kwargs):
    return ResNetMulpart(50, **kwargs)


def resnet101mulpart(**kwargs):
    return ResNetMulpart(101, **kwargs)


def resnet152mulpart(**kwargs):
    return ResNetMulpart(152, **kwargs)