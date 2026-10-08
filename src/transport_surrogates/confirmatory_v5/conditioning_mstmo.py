"""Regime-conditioned MS-TMO-UNet for the cross-architecture study.

The conditioning semantics match :class:`RegimeConditionedCTAUNet`: the
backbone receives only conductivity, the two normalized transport codes are
read from the spatially constant condition maps, and a compact factorized code
modulates the first encoder block through FiLM.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
import torch.nn as nn

from src.transport_surrogates.baselines.train_unet_multiout_logc_B_aspp_attn import (
    UNet_ASPP_Attn,
)
from src.transport_surrogates.confirmatory_v5.conditioning import (
    CONDITIONING_VARIANTS,
    _centers_tensor,
)


class RegimeConditionedMSTMO(UNet_ASPP_Attn):
    """MS-TMO-UNet with a compact regime code and early-feature FiLM."""

    def __init__(
        self,
        *,
        conditioning_variant: str,
        normalized_level_centers: Mapping[str, Sequence[float]],
        out_ch: int = 25,
        base: int = 64,
        attn_heads: int = 4,
        film_embed_dim: int = 8,
        film_hidden_dim: int = 32,
    ) -> None:
        if conditioning_variant == "v4_joint_continuous":
            raise ValueError("v4_joint_continuous is implemented for the CTA-UNet backbone only.")
        if conditioning_variant not in CONDITIONING_VARIANTS[1:]:
            raise ValueError(
                "RegimeConditionedMSTMO requires one of "
                f"{CONDITIONING_VARIANTS[1:]}, got {conditioning_variant!r}"
            )
        if film_embed_dim <= 0 or film_hidden_dim <= 0:
            raise ValueError("FiLM embedding and hidden dimensions must be positive.")
        super().__init__(in_ch=1, out_ch=out_ch, base=base, attn_heads=attn_heads)

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
        else:
            self.alpha_l_encoder = nn.Sequential(
                nn.Linear(1, film_hidden_dim),
                nn.SiLU(),
                nn.Linear(film_hidden_dim, film_embed_dim),
            )
            self.alpha_t_ratio_encoder = nn.Sequential(
                nn.Linear(1, film_hidden_dim),
                nn.SiLU(),
                nn.Linear(film_hidden_dim, film_embed_dim),
            )

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

    def regime_indices(
        self, normalized_codes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if normalized_codes.ndim != 2 or normalized_codes.shape[1] != 2:
            raise ValueError(
                "Expected conditioning scalars with shape [B,2], got "
                f"{tuple(normalized_codes.shape)}"
            )
        alpha_l_index = self._nearest_level(
            normalized_codes[:, 0], self.alpha_l_centers
        )
        ratio_index = self._nearest_level(
            normalized_codes[:, 1], self.alpha_t_ratio_centers
        )
        return alpha_l_index, ratio_index

    def encode_conditioning(self, normalized_codes: torch.Tensor) -> torch.Tensor:
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
                [
                    self.alpha_l_encoder(alpha_l_index),
                    self.alpha_t_ratio_encoder(ratio_index),
                ],
                dim=1,
            )
        return self.joint_encoder(alpha_l_index * 2 + ratio_index)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply FiLM after the first MS-TMO encoder block."""
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(
                "Expected [B,3,H,W] input with K and two condition maps, got "
                f"{tuple(x.shape)}"
            )
        normalized_codes = x[:, 1:3, 0, 0]
        code = self.encode_conditioning(normalized_codes)
        gamma, beta = self.film(code).chunk(2, dim=1)

        x1 = self.inc(x[:, 0:1])
        x1 = x1 * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)

        middle = self.mid(x4)
        middle = self.aspp(middle)
        middle = self.attn(middle)

        y = self.u3(middle, x3)
        y = self.u2(y, x2)
        y = self.u1(y, x1)
        return self.out(y)
