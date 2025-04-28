
import torch.nn as nn
import torch.nn.functional as F
from .common import Block


class ResNet(nn.Module):
    def __init__(self, num_classes=7):
        super(ResNet, self).__init__()

    def forward(self, x):
        return x
