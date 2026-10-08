import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        groups = 8 if out_ch % 8 == 0 else 1
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=True),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=True),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, 2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        dh = skip.shape[-2] - x.shape[-2]
        dw = skip.shape[-1] - x.shape[-1]
        if dh != 0 or dw != 0:
            x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class ASPP(nn.Module):
    def __init__(self, ch: int, rates=(1, 6, 12, 18)):
        super().__init__()
        groups = 8 if ch % 8 == 0 else 1
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch, 3, 1, padding=r, dilation=r, bias=True),
                nn.GroupNorm(groups, ch),
                nn.SiLU(inplace=True),
            )
            for r in rates
        ])
        self.proj = nn.Sequential(
            nn.Conv2d(ch * len(rates), ch, 1, 1, 0, bias=True),
            nn.GroupNorm(groups, ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([b(x) for b in self.branches], dim=1))


class BottleneckAttention(nn.Module):
    def __init__(self, ch: int, heads: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(ch)
        self.norm2 = nn.LayerNorm(ch)
        self.attn = nn.MultiheadAttention(ch, heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(ch, 4 * ch),
            nn.GELU(),
            nn.Linear(4 * ch, ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, ch, hgt, wid = x.shape
        s = x.permute(0, 2, 3, 1).reshape(bsz, hgt * wid, ch)
        s = s + self.attn(self.norm1(s), self.norm1(s), self.norm1(s), need_weights=False)[0]
        s = s + self.ff(self.norm2(s))
        return s.reshape(bsz, hgt, wid, ch).permute(0, 3, 1, 2)


class PMT_UNet_ASPP_Attn(nn.Module):
    """
    Physics-guided multi-task Option-B variant.
    Main output: log10 concentration maps for all timesteps.
    Aux outputs:
      - plume-mask logits for all timesteps
      - total-mass logits per timestep from global bottleneck features
    """

    def __init__(self, in_ch: int = 1, out_ch: int = 25, base: int = 64, attn_heads: int = 4):
        super().__init__()
        b = base
        self.out_ch = out_ch

        self.inc = ConvBlock(in_ch, b)
        self.d1 = Down(b, 2 * b)
        self.d2 = Down(2 * b, 4 * b)
        self.d3 = Down(4 * b, 8 * b)

        self.mid = ConvBlock(8 * b, 8 * b)
        self.aspp = ASPP(8 * b)
        self.attn = BottleneckAttention(8 * b, heads=attn_heads)

        self.u3 = Up(8 * b, 4 * b)
        self.u2 = Up(4 * b, 2 * b)
        self.u1 = Up(2 * b, b)

        self.map_head = nn.Conv2d(b, out_ch, 1, 1, 0)
        self.plume_head = nn.Sequential(
            ConvBlock(b, b),
            nn.Conv2d(b, out_ch, 1, 1, 0),
        )
        self.mass_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8 * b, 4 * b),
            nn.GELU(),
            nn.Linear(4 * b, out_ch),
        )

    def _forward_impl(self, x: torch.Tensor):
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)

        m = self.mid(x4)
        m = self.aspp(m)
        m = self.attn(m)

        y = self.u3(m, x3)
        y = self.u2(y, x2)
        y = self.u1(y, x1)

        pred_map = self.map_head(y)
        pred_plume = self.plume_head(y)
        pred_mass = self.mass_head(m)
        return pred_map, pred_plume, pred_mass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pred_map, _, _ = self._forward_impl(x)
        return pred_map

    def forward_with_aux(self, x: torch.Tensor):
        return self._forward_impl(x)
