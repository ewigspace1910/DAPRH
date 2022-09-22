from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch
from . layers import MyBatchNorm1D, Bottleneck, conv1x1, conv3x3
from .orginal import osnet_ibn_x1_0, osnet_x1_0, osnet_x0_25, osnet_x0_5, osnet_x0_75

__all__ = [  "OSNet", "osnet0_25", "osnet0_5",   "osnet0_75",   "osnet1_0",   "osnet1_0ibt" ]

class OSNet(nn.Module):
    __factory = {
        "1_0": osnet_x1_0,
        "1_0ibn": osnet_ibn_x1_0,
        "0_75": osnet_x0_75,
        "0_5": osnet_x0_5,
        "0_25": osnet_x0_25       
    }

    def __init__(self, depth, cfg, num_classes, num_features=0, dropout=0, pretrained=True,
                is_export=False, **kwargs):
        super(OSNet, self).__init__()
        #for onnx
        self.my_norm = MyBatchNorm1D()
        self.is_export = is_export

        self.pretrained = cfg.MODEL.BACKBONE.PRETRAIN
        self.depth = depth
        self.num_features = num_features
        self.has_embedding = num_features > 0
        self.dropout = dropout
        # Construct base (pretrained) model
        if depth not in OSNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        model = OSNet.__factory[depth]()

        self.base = nn.Sequential(model.conv1, model.maxpool, model.conv2, model.conv3,
                            model.conv4, model.conv5) #ignore last relu layer 
        self.gap = nn.AdaptiveAvgPool2d(1)

        # self.num_features = resnet.fc.in_features
        self.num_classes = num_classes
        out_planes = model.classifier.in_features

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
        planes = 128
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

    def forward(self, x, finetune=False):
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
    return OSNet("1_0ibn", **kwargs)