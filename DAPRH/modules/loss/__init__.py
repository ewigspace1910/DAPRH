from __future__ import absolute_import

from .triplet import TripletLoss, SoftTripletLoss
from .crossentropy import CrossEntropyLabelSmooth, SoftEntropy, KLDivLoss, CrossEntropyLabelSmoothFilterNoise
from .extra import UET
from .partavgtriplet import PartAveragedTripletLoss
from .center_triplet import CenterTripletLoss
__all__ = [
    'TripletLoss',
    'CrossEntropyLabelSmooth',
    'SoftTripletLoss',
    'SoftEntropy',
    'KLDivLoss',
    'CrossEntropyLabelSmoothFilterNoise',
    "UET",
    "RegLoss",
    "PartAveragedTripletLoss",
    "CenterTripletLoss"
    
]
