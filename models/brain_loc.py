"""
Brain-loc module
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_blocks import BasicConvBlock


class LocUnet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, bilinear_mode=False):
        super().__init__()

        self.in_conv = BasicConvBlock(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.up1 = Up(512, 256, bilinear_mode)
        self.up2 = Up(256, 128, bilinear_mode)
        self.up3 = Up(128, 64, bilinear_mode)
        self.out_conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=(3, 3), padding=(1, 1))
        )

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.out_conv(x)

        return x


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            BasicConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up_tran = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.db_conv = BasicConvBlock(in_channels, out_channels, in_channels // 2)
        else:
            self.up_tran = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=(2, 2), stride=(2, 2))
            self.db_conv = BasicConvBlock(in_channels, out_channels)

    def forward(self, x_up, x_down):
        x_up = self.up_tran(x_up)

        diffX = x_down.size()[3] - x_up.size()[3]
        diffY = x_down.size()[2] - x_up.size()[2]

        x_up = F.pad(x_up, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        x = torch.cat((x_down, x_up), dim=1)
        x = self.db_conv(x)

        return x
