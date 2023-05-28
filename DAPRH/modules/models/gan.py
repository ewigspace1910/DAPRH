import torch.nn as nn
import torch.nn.functional as F

class DisNet(nn.Module):
    def __init__(self, input=2048):
        super(DisNet, self).__init__()
        self.bn0 = nn.BatchNorm1d(input)
        self.fc1 = nn.Linear(input, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 1)
        nn.init.normal_(self.fc1.weight, std=0.001)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight, std=0.001)
        nn.init.constant_(self.fc2.bias, 0)
    def forward(self, x):
        x = self.fc1(self.bn0(x))
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x