import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = 8 if channels % 8 == 0 else 1
    return nn.GroupNorm(groups, channels)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            _group_norm(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            _group_norm(out_ch),
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
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
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
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=rate, dilation=rate, bias=True),
                    _group_norm(ch),
                    nn.SiLU(inplace=True),
                )
                for rate in rates
            ]
        )
        self.proj = nn.Sequential(
            nn.Conv2d(ch * len(rates), ch, kernel_size=1, bias=True),
            _group_norm(ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([branch(x) for branch in self.branches], dim=1))


class BottleneckAttention(nn.Module):
    def __init__(self, ch: int, heads: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(ch)
        self.norm2 = nn.LayerNorm(ch)
        self.attn = nn.MultiheadAttention(embed_dim=ch, num_heads=heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(ch, ch * 4),
            nn.GELU(),
            nn.Linear(ch * 4, ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, channels, height, width = x.shape
        seq = x.flatten(2).transpose(1, 2)
        attn_in = self.norm1(seq)
        seq = seq + self.attn(attn_in, attn_in, attn_in, need_weights=False)[0]
        seq = seq + self.ffn(self.norm2(seq))
        return seq.transpose(1, 2).view(bsz, channels, height, width)


class CrossTimestepAttention(nn.Module):
    def __init__(self, timesteps: int = 25, embed_dim: int = 25, heads: int = 5, pool_stride: int = 4):
        super().__init__()
        if embed_dim % heads != 0:
            raise ValueError("embed_dim must be divisible by heads")
        self.pool_stride = pool_stride
        self.token_embed = nn.Linear(1, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.token_proj = nn.Linear(embed_dim, 1)
        nn.init.zeros_(self.token_proj.weight)
        nn.init.zeros_(self.token_proj.bias)
        self.timesteps = timesteps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, timesteps, height, width = x.shape
        if timesteps != self.timesteps:
            raise ValueError(f"Expected {self.timesteps} timesteps, got {timesteps}")

        pooled = F.avg_pool2d(x, self.pool_stride, self.pool_stride)
        _, _, pooled_h, pooled_w = pooled.shape
        tokens = pooled.permute(0, 2, 3, 1).contiguous().view(bsz * pooled_h * pooled_w, timesteps, 1)
        tokens = self.token_embed(tokens)
        attn_in = self.norm1(tokens)
        tokens = tokens + self.attn(attn_in, attn_in, attn_in, need_weights=False)[0]
        tokens = tokens + self.ffn(self.norm2(tokens))
        out = self.token_proj(tokens).squeeze(-1)
        out = out.view(bsz, pooled_h, pooled_w, timesteps).permute(0, 3, 1, 2)
        out = F.interpolate(out, size=(height, width), mode="bilinear", align_corners=False)
        return x + out


class CTAUNet(nn.Module):
    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = 25,
        base: int = 64,
        t_embed_dim: int = 25,
        t_heads: int = 5,
        pool_stride: int = 4,
    ):
        super().__init__()
        b = base
        self.enc0 = ConvBlock(in_ch, b)
        self.enc1 = Down(b, b * 2)
        self.enc2 = Down(b * 2, b * 4)
        self.enc3 = Down(b * 4, b * 8)
        self.bottleneck_conv = ConvBlock(b * 8, b * 8)
        self.aspp = ASPP(b * 8)
        self.bottleneck_attn = BottleneckAttention(b * 8, heads=4)
        self.up3 = Up(b * 8, b * 4)
        self.up2 = Up(b * 4, b * 2)
        self.up1 = Up(b * 2, b)
        self.out_conv = nn.Conv2d(b, out_ch, kernel_size=1)
        self.temporal_attn = CrossTimestepAttention(
            timesteps=out_ch,
            embed_dim=t_embed_dim,
            heads=t_heads,
            pool_stride=pool_stride,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc0 = self.enc0(x)
        enc1 = self.enc1(enc0)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        bottleneck = self.bottleneck_conv(enc3)
        bottleneck = self.aspp(bottleneck)
        bottleneck = self.bottleneck_attn(bottleneck)
        dec = self.up3(bottleneck, enc2)
        dec = self.up2(dec, enc1)
        dec = self.up1(dec, enc0)
        out = self.out_conv(dec)
        return self.temporal_attn(out)
