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


class Bottleneck(nn.Module):
    def __init__(self, in_ch, mid_ch, exp, is_bottle, stride):
        super().__init__()
        self.expansion = exp
        self.residual = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch*exp, 1, bias=False),
            nn.BatchNorm2d(mid_ch*exp),
        )
        if is_bottle or in_ch != mid_ch*exp:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, mid_ch*exp, 1, stride=stride, bias=False),
                nn.BatchNorm2d(mid_ch*exp)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.residual(x)
        out += self.shortcut(x)
        return F.relu(out)
