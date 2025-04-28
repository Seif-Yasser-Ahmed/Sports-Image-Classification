import torch.nn as nn
import torch.nn.functional as F
from .common import Block


class VGG16(nn.Module):
    def __init__(self, num_classes=7):
        super(VGG16, self).__init__()

    def forward(self, x):
        return x
