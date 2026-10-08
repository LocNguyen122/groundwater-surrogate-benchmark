import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        return x, self.pool(x)


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet2D_TimeCond_MultiHead(nn.Module):
    """
    Time-conditioned UNet with two heads:
      - main head: log10(C+eps) map (B,1,H,W)
      - aux head : patch mass scalar (B,1)

    Input: (B,2,H,W) = [K_norm, t_norm]
    """
    def __init__(self, in_ch=2, base=64, mass_mlp_hidden=256):
        super().__init__()
        b = base

        self.d1 = Down(in_ch, b)
        self.d2 = Down(b, 2 * b)
        self.d3 = Down(2 * b, 4 * b)

        self.mid = ConvBlock(4 * b, 8 * b)

        self.u3 = Up(8 * b, 4 * b, 4 * b)
        self.u2 = Up(4 * b, 2 * b, 2 * b)
        self.u1 = Up(2 * b, b, b)

        # Main map head
        self.out_map = nn.Conv2d(b, 1, kernel_size=1)

        # Aux mass head from bottleneck features (global context)
        self.mass_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),     # (B,8b,1,1)
            nn.Flatten(),               # (B,8b)
            nn.Linear(8 * b, mass_mlp_hidden),
            nn.SiLU(inplace=True),
            nn.Linear(mass_mlp_hidden, 1),
        )

    def forward(self, x):
        s1, x = self.d1(x)
        s2, x = self.d2(x)
        s3, x = self.d3(x)

        x = self.mid(x)

        mass_pred = self.mass_head(x)  # (B,1)

        x = self.u3(x, s3)
        x = self.u2(x, s2)
        x = self.u1(x, s1)

        y = self.out_map(x)  # (B,1,H,W)
        return y, mass_pred