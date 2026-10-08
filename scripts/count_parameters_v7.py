"""Parameter counts of every model configuration used in the manuscript (CPU, no data needed).

Usage: python scripts/count_parameters_v7.py --out results/tables/parameter_counts_v7.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.transport_surrogates.baselines.fno2d import FNO2D
from src.transport_surrogates.baselines.temporal_attn_unet import CTAUNet
from src.transport_surrogates.baselines.train_unet_multiout_logc_B_aspp_attn import UNet_ASPP_Attn
from src.transport_surrogates.confirmatory_v5.conditioning import RegimeConditionedCTAUNet

CENTERS = {"alpha_L": [-1.118, 0.0, 1.118], "alpha_T_ratio": [-1.0, 1.0]}


def n(m) -> int:
    return sum(p.numel() for p in m.parameters())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = [("cta_unet_k_only", n(CTAUNet(in_ch=1, out_ch=25, base=64, t_embed_dim=25, t_heads=5, pool_stride=4))),
            ("cta_unet_v0_raw_channels", n(CTAUNet(in_ch=3, out_ch=25, base=64, t_embed_dim=25, t_heads=5, pool_stride=4))),
            ("ms_tmo_v0_raw_channels", n(UNet_ASPP_Attn(in_ch=3, out_ch=25, base=64, attn_heads=4))),
            ("fno_v0_raw_channels", n(FNO2D(in_ch=3, out_ch=25, width=64, modes1=20, modes2=20, depth=4)))]
    for v in ("v1_factorized_levels", "v2_joint_embedding", "v3_factorized_continuous", "v4_joint_continuous"):
        rows.append((f"cta_unet_{v}", n(RegimeConditionedCTAUNet(conditioning_variant=v, normalized_level_centers=CENTERS,
                                                               out_ch=25, base=64, t_embed_dim=25, t_heads=5, pool_stride=4,
                                                               film_embed_dim=8, film_hidden_dim=32))))
    base = dict(rows)["cta_unet_k_only"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["configuration", "parameters", "added_vs_cta_k_only"])
        for name, p in rows:
            w.writerow([name, p, p - base])
            print(f"{name:32s} {p:>12,d} {p - base:>+8,d}")


if __name__ == "__main__":
    main()
