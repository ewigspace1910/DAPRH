import torch.nn.functional as F

def make_loss(Cfg, num_classes):    # modified by gu
    return loss_func