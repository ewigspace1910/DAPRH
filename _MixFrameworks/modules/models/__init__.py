from __future__ import absolute_import

from .part.regnet_part import regnetY128gf_part, regnetY16gf_part, regnetY1_6gf_part, regnetY32gf_part, regnetY3_2gf_part, regnetY400_part, regnetY800_part
from .part.mnasnet_part import *
from .part.osnet_part import *
from .part.resnet_part import *
from .part.mobilenet_part import  mobileNetLpart, mobileNetSpart

from .mobilenet import mobileNetL, mobileNetS
from .mnasnet import *
from .regnet import *
from .osnet import *
from .resnet import *
from .resnet_ibn import *


__factory = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'resnet_ibn50a': resnet_ibn50a,
    'resnet_ibn101a': resnet_ibn101a,
    'resnet18part': resnet18part,
    'resnet34part': resnet34part,
    'resnet50part': resnet50part,
    'resnet101part': resnet101part,
    'resnet152part': resnet152part,

    #mobile net
    'mobilenetS': mobileNetS,
    'mobilenetL': mobileNetL,
    'mobilenetSpart': mobileNetSpart,
    'mobilenetLpart': mobileNetLpart,

    #RegNet
    "regnetY128gf"  : regnetY128gf,
    "regnetY32gf"   : regnetY32gf,
    "regnetY16gf"   : regnetY16gf,
    "regnetY3_2gf"  : regnetY3_2gf,
    "regnetY1_6gf"  : regnetY1_6gf,
    "regnetY400"    : regnetY400,
    "regnetY800"    : regnetY800,
    "regnetY128gf_part"  : regnetY128gf_part,
    "regnetY32gf_part"   : regnetY32gf_part,
    "regnetY16gf_part"   : regnetY16gf_part,
    "regnetY3_2gf_part"  : regnetY3_2gf_part,
    "regnetY1_6gf_part"  : regnetY1_6gf_part,
    "regnetY400_part"    : regnetY400_part,
    "regnetY800_part"    : regnetY800_part,

    #MNastNet
    "mnasnet0_5": mnasnet0_5,
    "mnasnet0_75": mnasnet0_75,
    "mnasnet1_0": mnasnet1_0,
    "mnasnet1_3": mnasnet1_3,
    "mnasnet0_5part": mnasnet0_5part,
    "mnasnet0_75part": mnasnet0_75part,
    "mnasnet1_0part": mnasnet1_0part,
    "mnasnet1_3part": mnasnet1_3part,

    #OSnet
    "osnet0_5"      : osnet0_5,   
    "osnet0_75"     : osnet0_75,  
    "osnet1_0"      : osnet1_0,   
    "osnet1_0ibt"   : osnet1_0ibt,
    "osnet0_5part"      : osnet0_5part,   
    "osnet0_75part"     : osnet0_75part,  
    "osnet1_0part"      : osnet1_0part,   
    "osnet1_0ibtpart"   : osnet1_0ibtpart,

    "":None
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """
    Create a model instance.

    Parameters
    ----------
    name : str
        Model name. Can be one of 'inception', 'resnet18', 'resnet34',
        'resnet50', 'resnet101', and 'resnet152'.
    pretrained : bool, optional
        Only applied for 'resnet*' models. If True, will use ImageNet pretrained
        model. Default: True
    cut_at_pooling : bool, optional
        If True, will cut the model before the last global pooling layer and
        ignore the remaining kwargs. Default: False
    num_features : int, optional
        If positive, will append a Linear layer after the global pooling layer,
        with this number of output units, followed by a BatchNorm layer.
        Otherwise these layers will not be appended. Default: 256 for
        'inception', 0 for 'resnet*'
    norm : bool, optional
        If True, will normalize the feature to be unit L2-norm for each sample.
        Otherwise will append a ReLU layer after the above Linear layer if
        num_features > 0. Default: False
    dropout : float, optional
        If positive, will append a Dropout layer with this dropout rate.
        Default: 0
    num_classes : int, optional
        If positive, will append a Linear layer at the end as the classifier
        with this number of output units. Default: 0
    """
    if name not in __factory:
        raise KeyError("Unknown model:", name)
    return __factory[name](*args, **kwargs)
