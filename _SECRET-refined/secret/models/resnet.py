from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch
from . layers import MyBatchNorm1D, Bottleneck, conv1x1, conv3x3
__all__ = ['resnet50', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']



class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, cfg, num_classes, num_features=0, dropout=0, pretrained=True,
                is_export=False):
        super(ResNet, self).__init__()

        self.pretrained = cfg.MODEL.BACKBONE.PRETRAIN
        self.depth = depth
        self.num_features = num_features
        self.has_embedding = num_features > 0
        self.dropout = dropout
        # Construct base (pretrained) resnet
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        if depth >= 50:
            resnet = ResNet.__factory[depth](pretrained=="IMAGENET1K_V2") #(pretrained=pretrained)
        else:
            resnet = ResNet.__factory[depth](pretrained=pretrained)
        if depth >= 50:
            resnet.layer4[0].conv2.stride = (1,1)
            resnet.layer4[0].downsample[0].stride = (1,1)
        self.base = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # self.num_features = resnet.fc.in_features
        self.num_classes = num_classes
        out_planes = resnet.fc.in_features

        # Append new layers
        self.part_detach = cfg.MODEL.PART_DETACH
        # Append new layers
        if self.has_embedding:
            self.feat = nn.Linear(out_planes, self.num_features)
            self.feat_bn = nn.BatchNorm1d(self.num_features)
            init.kaiming_normal_(self.feat.weight, mode='fan_out')
            init.constant_(self.feat.bias, 0)
        else:
            self.num_features = out_planes
            self.feat_bn = nn.BatchNorm1d(self.num_features)
        self.feat_bn.bias.requires_grad_(False)
        init.constant_(self.feat_bn.weight, 1)
        init.constant_(self.feat_bn.bias, 0)

        self.classifier = nn.Linear(self.num_features, self.num_classes, bias=False)
        init.normal_(self.classifier.weight, std=0.001)

        #####part#########
        norm_layer = nn.BatchNorm2d
        block = Bottleneck
        planes = 512
        if self.has_embedding:
            self.part_num_features = num_features
        else:
            self.part_num_features = planes * block.expansion

        downsample = nn.Sequential(
            conv1x1(out_planes, block.expansion * planes),
            norm_layer(block.expansion * planes),
        )
        self.part_bottleneck = block(
                out_planes, planes, downsample = downsample, norm_layer = norm_layer
            )

        self.part_pool = nn.AdaptiveAvgPool2d((2,1))

        if self.has_embedding:
            self.part_num_features = self.num_features
            self.partup_feat = nn.Linear(out_planes, self.num_features, bias=False)
            self.partdown_feat = nn.Linear(out_planes, self.num_features, bias=False)
        else:
            
            # Change the num_features to CNN output channels
            self.num_features = self.part_num_features

        self.partup_feat_bn = nn.BatchNorm1d(self.part_num_features)
        self.partup_feat_bn.bias.requires_grad_(False)
        init.constant_(self.partup_feat_bn.weight, 1)
        init.constant_(self.partup_feat_bn.bias, 0)

        self.partdown_feat_bn = nn.BatchNorm1d(self.part_num_features)
        self.partdown_feat_bn.bias.requires_grad_(False)
        init.constant_(self.partdown_feat_bn.weight, 1)
        init.constant_(self.partdown_feat_bn.bias, 0)
            #classifier for part
        self.classifier_partup = nn.Linear(self.part_num_features, self.num_classes, bias = False)
        init.normal_(self.classifier_partup.weight, std=0.001)
        self.classifier_partdown = nn.Linear(self.part_num_features, self.num_classes, bias = False)
        init.normal_(self.classifier_partdown.weight, std=0.001)

        if not self.pretrained:
            self.reset_params()

    def forward(self, x, finetune = False):
        featuremap = self.base(x)

        x = self.gap(featuremap)
        x = x.view(x.size(0), -1)

        if self.has_embedding:
            bn_x = self.feat_bn(self.feat(x))
        else:
            bn_x = self.feat_bn(x) #[bs, 2048] #global feature
        if self.training is False and finetune is False:
            bn_x = F.normalize(bn_x)
            return [bn_x]
        ##########part#############
        if self.part_detach:
            part_x = self.part_bottleneck(featuremap.detach())
        else:
            part_x = self.part_bottleneck(featuremap)

        part_x = self.part_pool(part_x)

        part_up = part_x[:, :, 0, :]
        part_down = part_x[:, :, 1, :]

        part_up = part_up.view(part_up.size(0), -1)
        part_down = part_down.view(part_down.size(0), -1)

        if self.has_embedding:
            part_up = self.partup_feat(part_up)
            part_down = self.partdown_feat(part_down)

        bn_part_up = self.partup_feat_bn(part_up)  
        bn_part_down = self.partdown_feat_bn(part_down)


        prob = self.classifier(bn_x)
        prob_part_up = self.classifier_partup(bn_part_up)
        prob_part_down = self.classifier_partdown(bn_part_down)

        if finetune is True:
            bn_x = F.normalize(bn_x)
            bn_part_up = F.normalize(bn_part_up)
            bn_part_down = F.normalize(bn_part_down)
            return [x, part_up, part_down], [bn_x, bn_part_up, bn_part_down], [prob, prob_part_up, prob_part_down]
        else:
            return [x, part_up, part_down], [prob, prob_part_up, prob_part_down]

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

        resnet = ResNet.__factory[self.depth](pretrained=self.pretrained)
        self.base[0].load_state_dict(resnet.conv1.state_dict())
        self.base[1].load_state_dict(resnet.bn1.state_dict())
        self.base[2].load_state_dict(resnet.relu.state_dict())
        self.base[3].load_state_dict(resnet.maxpool.state_dict())
        self.base[4].load_state_dict(resnet.layer1.state_dict())
        self.base[5].load_state_dict(resnet.layer2.state_dict())
        self.base[6].load_state_dict(resnet.layer3.state_dict())
        self.base[7].load_state_dict(resnet.layer4.state_dict())

def resnet18(**kwargs):
    return ResNet(18, **kwargs)


def resnet34(**kwargs):
    return ResNet(34, **kwargs)


def resnet50(**kwargs):
    return ResNet(50, **kwargs)


def resnet101(**kwargs):
    return ResNet(101, **kwargs)


def resnet152(**kwargs):
    return ResNet(152, **kwargs)
