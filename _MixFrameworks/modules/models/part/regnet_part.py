from __future__ import absolute_import
from this import d

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch

__all__ = [
    "Regnetpart",
    "regnetY128gf_part",
    "regnetY32gf_part",
    "regnetY16gf_part",
    "regnetY3_2gf_part",
    "regnetY1_6gf_part",
    "regnetY400_part",
    "regnetY800_part"
]

class Regnetpart(nn.Module):
    __factory = {
    "regnetY128gf_part"  : torchvision.models.regnet_y_128gf,
    "regnetY32gf_part"   : torchvision.models.regnet_y_32gf,
    "regnetY16gf_part"   : torchvision.models.regnet_y_16gf,
    "regnetY3_2gf_part"  : torchvision.models.regnet_y_3_2gf,
    "regnetY1_6gf_part"  : torchvision.models.regnet_y_1_6gf,
    "regnetY400_part"    : torchvision.models.regnet_y_400mf,
    "regnetY800_part"    : torchvision.models.regnet_y_800mf
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                num_parts=3, num_features=0, norm=False, dropout=0, num_classes=0, **kwargs):
        super(Regnetpart, self).__init__()
        self.pretrained = pretrained
        self.depth = depth
        self.cut_at_pooling = cut_at_pooling
        # Construct base (pretrained) model
        if depth not in Regnetpart.__factory:
            raise KeyError("Unsupported depth:", depth)
        try:
            model = Regnetpart.__factory[depth](weights="IMAGENET1K_SWAG_LINEAR_V1")
        except:
            model = Regnetpart.__factory[depth](weights="IMAGENET1K_V2")

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
        
        ##########################
        #ADD ideal in PPLR 
        #https://github.com/ewigspace1910/Paper-Notes-Deep-Learning/blob/main/Computer%20Vision/3.Person%20ReID/PPLR.md
        self.num_parts = num_parts
        self.rap = nn.AdaptiveAvgPool2d((self.num_parts, 1))

        # part feature classifiers
        for i in range(self.num_parts):
            # Append new layers
            if self.has_embedding:
                name = 'embedding' + str(i)
                setattr(self, name, nn.Linear(out_planes, self.num_features, bias=False))
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
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
        f_g = self.gap(x) # [bs, 2048, 1, 1]
        f_g = f_g.view(x.size(0), -1)

        if self.cut_at_pooling:
            return f_g
        if self.has_embedding:
            bn_x = self.feat_bn(self.feat(f_g))
        else:
            bn_x = self.feat_bn(f_g)
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
            if self.has_embedding:
                f_p_i = getattr(self, 'embedding' + str(i))(f_p_i)
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

        model = Regnetpart.__factory[self.depth](pretrained=self.pretrained)
        
        model = nn.Sequential(model.stem, model.trunk_output)
        self.base.load_state_dict(model.state_dict()) #check orginal model if get error

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
            if self.has_embedding:
                f_p_i = getattr(self, 'embedding' + str(i))(f_p_i)
            f_p_i = getattr(self, 'bnneck' + str(i))(f_p_i)
            f_p_i = F.normalize(f_p_i)
            fs_p.append(f_p_i)
        fs_p = torch.stack(fs_p, dim=-1)

        return f_g, fs_p

####------------------------------####
def regnetY128gf_part(**kwargs):
    return Regnetpart("regnetY128gf", **kwargs)

def regnetY32gf_part(**kwargs):
    return Regnetpart("regnetY32gf", **kwargs)

def regnetY16gf_part(**kwargs):
    return Regnetpart("regnetY16gf", **kwargs)

def regnetY3_2gf_part(**kwargs):
    return Regnetpart("regnetY3_2gf", **kwargs)

def regnetY1_6gf_part(**kwargs):
    return Regnetpart("regnetY1_6gf", **kwargs)

def regnetY400_part(**kwargs):
    return Regnetpart("regnetY400", **kwargs)

def regnetY800_part(**kwargs):
    return Regnetpart("regnetY800", **kwargs)




