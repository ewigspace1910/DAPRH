from __future__ import absolute_import

from .part.osnet_part import *
from .part.resnet_part import *
from .part.mobilenet_part import  mobileNetLpart, mobileNetSpart

from .mobilenet import mobileNetL, mobileNetS
from .osnet import *
from .resnet import *



__factory = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
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
