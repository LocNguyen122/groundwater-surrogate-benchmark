"""Conditioning modules for the isolated Obj1 compositional pilot.

The dataset continues to provide ``[K, alpha_L, alpha_T_ratio]`` maps.  The
factorized variants read the two spatially constant normalized codes and apply
a small FiLM modulation after the first CTA-UNet encoder block.  ``v4_joint_continuous``
is the injection-matched non-factorized control: identical FiLM site, shared network and
parameter budget, but a single joint encoder of both codes.  The backbone
and output head are otherwise unchanged.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
import torch.nn as nn

from src.transport_surrogates.baselines.temporal_attn_unet import CTAUNet


CONDITIONING_VARIANTS = (
    "v0_raw_channels",
    "v1_factorized_levels",
    "v2_joint_embedding",
    "v3_factorized_continuous",
    "v4_joint_continuous",
)


def _centers_tensor(
    centers: Mapping[str, Sequence[float]], name: str, expected: int
) -> torch.Tensor:
    values = torch.as_tensor(list(centers[name]), dtype=torch.float32)
    if values.ndim != 1 or values.numel() != expected:
        raise ValueError(f"Expected {expected} normalized centers for {name}, got {values.tolist()}")
    if not torch.all(values[1:] > values[:-1]):
        raise ValueError(f"Centers for {name} must be strictly increasing: {values.tolist()}")
    return values


class RegimeConditionedCTAUNet(CTAUNet):
    """CTA-UNet with a compact regime code and early-feature FiLM modulation."""

    def __init__(
        self,
        *,
        conditioning_variant: str,
        normalized_level_centers: Mapping[str, Sequence[float]],
        out_ch: int = 25,
        base: int = 64,
        t_embed_dim: int = 25,
        t_heads: int = 5,
        pool_stride: int = 4,
        film_embed_dim: int = 8,
        film_hidden_dim: int = 32,
    ) -> None:
        if conditioning_variant not in CONDITIONING_VARIANTS[1:]:
            raise ValueError(
                "RegimeConditionedCTAUNet requires one of "
                f"{CONDITIONING_VARIANTS[1:]}, got {conditioning_variant!r}"
            )
        if film_embed_dim <= 0 or film_hidden_dim <= 0:
            raise ValueError("FiLM embedding and hidden dimensions must be positive.")
        super().__init__(
            in_ch=1,
            out_ch=out_ch,
            base=base,
            t_embed_dim=t_embed_dim,
            t_heads=t_heads,
            pool_stride=pool_stride,
        )
        self.conditioning_variant = conditioning_variant
        self.film_embed_dim = int(film_embed_dim)
        self.film_hidden_dim = int(film_hidden_dim)
        self.register_buffer(
            "alpha_l_centers",
            _centers_tensor(normalized_level_centers, "alpha_L", expected=3),
        )
        self.register_buffer(
            "alpha_t_ratio_centers",
            _centers_tensor(normalized_level_centers, "alpha_T_ratio", expected=2),
        )

        if conditioning_variant == "v1_factorized_levels":
            self.alpha_l_encoder = nn.Embedding(3, film_embed_dim)
            self.alpha_t_ratio_encoder = nn.Embedding(2, film_embed_dim)
        elif conditioning_variant == "v2_joint_embedding":
            self.joint_encoder = nn.Embedding(6, 2 * film_embed_dim)
        elif conditioning_variant == "v3_factorized_continuous":
            self.alpha_l_encoder = nn.Sequential(
                nn.Linear(1, film_hidden_dim), nn.SiLU(), nn.Linear(film_hidden_dim, film_embed_dim)
            )
            self.alpha_t_ratio_encoder = nn.Sequential(
                nn.Linear(1, film_hidden_dim), nn.SiLU(), nn.Linear(film_hidden_dim, film_embed_dim)
            )
        elif conditioning_variant == "v4_joint_continuous":
            # Injection-matched non-factorized comparator. One joint MLP of both continuous
            # codes replaces the two marginal encoders; it emits the same 2 * film_embed_dim
            # vector, enters the same shared FiLM network, and modulates the same feature map.
            # Only the factorization differs, so V3 minus V4 isolates factorization from the
            # FiLM injection mechanism and location.
            self.joint_continuous_encoder = nn.Sequential(
                nn.Linear(2, film_hidden_dim),
                nn.SiLU(),
                nn.Linear(film_hidden_dim, 2 * film_embed_dim),
            )
        else:
            raise ValueError(f"Unknown conditioning_variant: {conditioning_variant!r}")

        self.film = nn.Sequential(
            nn.Linear(2 * film_embed_dim, film_hidden_dim),
            nn.SiLU(),
            nn.Linear(film_hidden_dim, 2 * base),
        )
        final = self.film[-1]
        nn.init.normal_(final.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(final.bias)

    @staticmethod
    def _nearest_level(values: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        return (values[:, None] - centers[None, :]).abs().argmin(dim=1)

    def regime_indices(self, normalized_codes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if normalized_codes.ndim != 2 or normalized_codes.shape[1] != 2:
            raise ValueError(f"Expected conditioning scalars with shape [B,2], got {normalized_codes.shape}")
        alpha_l_index = self._nearest_level(normalized_codes[:, 0], self.alpha_l_centers)
        ratio_index = self._nearest_level(normalized_codes[:, 1], self.alpha_t_ratio_centers)
        return alpha_l_index, ratio_index

    def encode_conditioning(self, normalized_codes: torch.Tensor) -> torch.Tensor:
        if self.conditioning_variant == "v4_joint_continuous":
            return self.joint_continuous_encoder(normalized_codes[:, 0:2])
        if self.conditioning_variant == "v3_factorized_continuous":
            return torch.cat(
                [
                    self.alpha_l_encoder(normalized_codes[:, 0:1]),
                    self.alpha_t_ratio_encoder(normalized_codes[:, 1:2]),
                ],
                dim=1,
            )
        alpha_l_index, ratio_index = self.regime_indices(normalized_codes)
        if self.conditioning_variant == "v1_factorized_levels":
            return torch.cat(
                [self.alpha_l_encoder(alpha_l_index), self.alpha_t_ratio_encoder(ratio_index)],
                dim=1,
            )
        return self.joint_encoder(alpha_l_index * 2 + ratio_index)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected [B,3,H,W] input with K and two condition maps, got {x.shape}")
        normalized_codes = x[:, 1:3, 0, 0]
        code = self.encode_conditioning(normalized_codes)
        gamma, beta = self.film(code).chunk(2, dim=1)

        enc0 = self.enc0(x[:, 0:1])
        enc0 = enc0 * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        enc1 = self.enc1(enc0)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        bottleneck = self.bottleneck_conv(enc3)
        bottleneck = self.aspp(bottleneck)
        bottleneck = self.bottleneck_attn(bottleneck)
        dec = self.up3(bottleneck, enc2)
        dec = self.up2(dec, enc1)
        dec = self.up1(dec, enc0)
        return self.temporal_attn(self.out_conv(dec))

