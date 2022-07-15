from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch

__all__ = [
    "Mnasnetpart",
    "mnasnet0_5part",
    "mnasnet0_75part",
    "mnasnet1_0part",
    "mnasnet1_3part",
]

class Mnasnetpart(nn.Module):
    __factory = {
        "0_5": torchvision.models.mnasnet0_5,
        "0_75": torchvision.models.mnasnet0_75,
        "1_0": torchvision.models.mnasnet1_0,
        "1_3": torchvision.models.mnasnet1_3    
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                num_parts=3, num_features=0, norm=False, dropout=0, num_classes=0):
        super(Mnasnetpart, self).__init__()
        self.pretrained = pretrained
        self.depth = depth
        self.cut_at_pooling = cut_at_pooling
        # Construct base (pretrained) model
        if depth not in Mnasnetpart.__factory:
            raise KeyError("Unsupported depth:", depth)
        model = Mnasnetpart.__factory[depth](weights="IMAGENET1K_V1")

        self.base = nn.Sequential(model.layers[:-1]) #ignore last relu layer 
        self.gap = nn.AdaptiveAvgPool2d(1)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes
            
            #  (classifier): Sequential((0): Dropout(p=0.2, inplace=True) (1): Linear(in_features=1280, out_features=1000, bias=True))
            out_planes = model.classifier[1].in_features #check orginal structure if get error!
            
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
        
        ##########################
        #ADD ideal in PPLR 
        #https://github.com/ewigspace1910/Paper-Notes-Deep-Learning/blob/main/Computer%20Vision/3.Person%20ReID/PPLR.md
        self.num_parts = num_parts
        self.rap = nn.AdaptiveAvgPool2d((self.num_parts, 1))

        # part feature classifiers
        for i in range(self.num_parts):
            name = 'bnneck' + str(i)
            setattr(self, name, nn.BatchNorm1d(self.num_features))
            init.constant_(getattr(self, name).weight, 1)
            init.constant_(getattr(self, name).bias, 0)
            getattr(self, name).bias.requires_grad_(False)

            name = 'classifier' + str(i)
            setattr(self, name, nn.Linear(self.num_features, self.num_classes, bias=False))

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

        ######################
        if self.norm:
            bn_x_ = F.normalize(bn_x)
        elif self.has_embedding:
            bn_x_ = F.relu(bn_x) 
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

        model = Mnasnetpart.__factory[self.depth](pretrained=self.pretrained)
        
        model = nn.Sequential(model.layers[:-1])
        self.base.load_state_dict(model.state_dict()) #check orginal model if get error

    def extract_all_features(self, x):
        x = self.base(x)
        f_g = self.gap(x)
        f_g = f_g.view(x.size(0), -1)
        if self.has_embedding:
            bn_x = self.feat_bn(self.feat(f_g))
        else:
            bn_x = self.feat_bn(f_g) #[bs, channel]
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

####------------------------------####
def mnasnet0_5part(**kwargs):
    return Mnasnetpart("0_5", **kwargs)

def mnasnet0_75part(**kwargs):
    return Mnasnetpart("0_75", **kwargs)

def mnasnet1_0part(**kwargs):
    return Mnasnetpart("1_0", **kwargs)

def mnasnet1_3part(**kwargs):
    return Mnasnetpart("1_3", **kwargs)




