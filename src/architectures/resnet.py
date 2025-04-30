import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import Block, Bottleneck
from ..utils.yaml import Config

cfg = Config.load()

NUM_CLASSES = cfg['NUM_CLASSES']


class ResNet(nn.Module):
    def __init__(self, num_classes=7):
        super(ResNet, self).__init__()

    def forward(self, x):
        return x

# Cell 8: ResNet50 definition


class ResNet50(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.layer1 = self._make_layer(64,  64, blocks=3, stride=1, exp=4)
        self.layer2 = self._make_layer(256, 128, blocks=4, stride=2, exp=4)
        self.layer3 = self._make_layer(512, 256, blocks=6, stride=2, exp=4)
        self.layer4 = self._make_layer(1024, 512, blocks=3, stride=2, exp=4)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*4, num_classes)

    def _make_layer(self, in_ch, out_ch, blocks, stride, exp):
        layers = [Bottleneck(in_ch, out_ch, exp, True, stride)]
        for _ in range(1, blocks):
            layers.append(Bottleneck(out_ch*exp, out_ch, exp, False, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
