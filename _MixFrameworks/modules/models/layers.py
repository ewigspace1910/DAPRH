from torch import nn
import torch

class MyBatchNorm1D(nn.Module):
    def __init__(self):
        super(MyBatchNorm1D, self).__init__()

    def forward(self, x, base_bn):
        mean = base_bn.running_mean.data
        var = base_bn.running_var.data
        weight = base_bn.weight.data
        bias = base_bn.bias.data

        y = (((x-mean)/torch.sqrt(var))*weight)+bias

        return y
