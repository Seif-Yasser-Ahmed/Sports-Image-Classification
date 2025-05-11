import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, act='relu', drop=False, norm=True):
        super().__init__()
        self.drop = drop
        self.norm = norm
        self.dropout = nn.Dropout(0.5)
        if self.norm == True:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 4, 2, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU() if act == 'relu' else nn.LeakyReLU(0.2),
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 4, 2, 1),
                nn.ReLU() if act == 'relu' else nn.LeakyReLU(0.2),
            )

    def forward(self, x):
        x = self.block(x)
        return self.dropout(x) if self.drop else x


class CustomCNN(nn.Module):
    def __init__(self, num_classes=7, input_channels=3, input_size=(224, 224), dropout=0.5, hidden_dim=64):
        super(CustomCNN, self).__init__()
        self.conv1 = Block(input_channels, hidden_dim, act='relu',
                           drop=False, norm=False)  # 3x224x224 -> 64x112x112
        self.maxpool1 = nn.MaxPool2d(
            kernel_size=2, stride=2)  # 64x112x112 -> 64x56x56
        self.conv2 = Block(hidden_dim, hidden_dim, act='relu',
                           drop=False, norm=False)  # 64x56x56 -> 64x28x28
        self.maxpool2 = nn.MaxPool2d(
            kernel_size=2, stride=2)  # 64x28x28 -> 64x14x14
        self.conv3 = Block(hidden_dim, hidden_dim*2, act='relu',
                           drop=False, norm=False)  # 64x14x14 -> 128x7x17
        self.maxpool3 = nn.MaxPool2d(
            kernel_size=2, stride=2)  # 128x7x7 -> 128x3x3
        self.conv4 = Block(hidden_dim*2, hidden_dim*2,
                           act='relu', drop=False, norm=False)
        self.maxpool4 = nn.MaxPool2d(
            kernel_size=2, stride=2)  # 128x3x3 -> 128x1x1
        self.fc1 = nn.Linear(128, 128)  # 128 -> 128
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)  # 128 -> num_classes
        self.sigmoid = nn.Sigmoid()
        # self.fc3 = nn.Linear(128*3*3, 128)  # 128 -> num_classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        # x = self.conv1(x)
        # x = self.maxpool1(x)
        # x = self.conv2(x)
        # x = self.maxpool2(x)
        # x = self.conv3(x)
        # x = self.maxpool3(x)
        # x=self.fc3(x.view(x.size(0), -1))
        # x = self.relu(x)
        # x = self.dropout(x)
        # x = self.fc2(x)

        return x
