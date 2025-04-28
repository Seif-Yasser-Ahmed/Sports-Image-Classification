import torch.nn as nn


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
