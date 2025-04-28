import torch.nn as nn
import torch.nn.functional as F
from .common import Block


class CustomCNN(nn.Module):
    def __init__(self, num_classes=7, input_channels=3, input_size=(224, 224), dropout=0.5, hidden_dim=64):
        super(CustomCNN, self).__init__()
        self.conv1 = Block(input_channels, hidden_dim, act='relu',
                           drop=False, norm=False)
        self.maxpool1 = nn.MaxPool2d(
            kernel_size=2, stride=2)
        self.conv2 = Block(hidden_dim, hidden_dim, act='relu',
                           drop=False, norm=False)
        self.maxpool2 = nn.MaxPool2d(
            kernel_size=2, stride=2)
        self.conv3 = Block(hidden_dim, hidden_dim*2,
                           act='relu', drop=False, norm=False)
        self.maxpool3 = nn.MaxPool2d(
            kernel_size=2, stride=2)
        self.conv4 = Block(hidden_dim*2, hidden_dim*2, act='relu',
                           drop=False, norm=False)
        self.maxpool4 = nn.MaxPool2d(
            kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)  # 3x224x224 -> 64x112x112
        x = self.maxpool1(x)  # 64x112x112 -> 64x56x56
        x = self.conv2(x)  # 64x56x56 -> 64x28x28
        x = self.maxpool2(x)  # 64x28x28 -> 64x14x14
        x = self.conv3(x)  # 64x14x14 -> 128x7x7
        x = self.maxpool3(x)  # 128x7x7 -> 128x3x3
        x = self.conv4(x)  # 128x3x3 -> 128x1x1
        x = x.view(x.size(0), -1)
        x = self.fc1(x)  # 128x1x1 -> 128
        x = self.relu(x)  # 128 -> 128
        x = self.dropout(x)  # 128 -> 128
        x = self.fc2(x)  # 128 -> num_classes

        return x
