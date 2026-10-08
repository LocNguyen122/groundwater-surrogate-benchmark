import torch
import torch.nn as nn
from .blocks import ConvBlock


class UpSampleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.up(x))


class UNet2D(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 25, base: int = 64):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base * 4, base * 8)

        # was ConvTranspose2d -> now Upsample+Conv
        self.up3 = UpSampleConv(base * 8, base * 4)
        self.dec3 = ConvBlock(base * 8, base * 4)

        self.up2 = UpSampleConv(base * 4, base * 2)
        self.dec2 = ConvBlock(base * 4, base * 2)

        self.up1 = UpSampleConv(base * 2, base)
        self.dec1 = ConvBlock(base * 2, base)

        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out(d1)