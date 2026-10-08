"""Figure: one geological setting, a twinned original test field versus an independent Pool T field.

Case rule (fixed before plotting): the first held-out geological cell in sorted (sigma2Y, correlation length,
anisotropy) order by recorded geology, transport setting (alphaL 6.2, ratio 0.1), realization 1 in both
populations, matched-code CTA-UNet seed 0 (best.pt), day 365. Rows: simulator, surrogate, setting mean (no K).
Predictions are produced with the unified evaluator's functions; plotted in the model-grid orientation (vertical
mirror of the stored array) like the other plume figures. Full 600x400 domain, no crop; display range fixed to
[-8, 0] in log10 C/C0 for every panel. The simulator plume outline (C = 1e-3 C0) is drawn on the surrogate and
setting-mean panels. A provenance JSON (checkpoint, statistics and case-file SHA-256, display range) and the
case's percentile among all per-case plume SSIM values of the same checkpoint are written next to the PDF.
"""
from __future__ import annotations

import argparse
import hashlib
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.shared.data.concentration import LOG10_MAX, LOG10_MIN
from src.shared.data.transport_conditioning import load_transport_metadata
from src.shared.eval import eval_transport_unified_logc_global_plumemask as EV

DAY_IDX = 24  # day 365
VMIN, VMAX, OUTLINE = -8.0, 0.0, -3.0


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def percentile(csv_path: str, folder: str, real: str) -> dict:
    import pandas as pd
    df = pd.read_csv(csv_path)
    v = float(df[(df.param_folder == folder) & (df.realization == real)].plume_ssim.iloc[0])
    return {"case_plume_ssim": v, "percentile": float((df.plume_ssim < v).mean() * 100), "n_cases": int(len(df))}


def predict(ckpt, stats, meta, path, device):
    cfg = ckpt["cfg"]
    model, _, _ = EV.build_model_from_ckpt(ckpt, device)
    b = cfg.get("output_bounds", [LOG10_MIN, LOG10_MAX])
    model = EV._BoundedLog10Output(model, float(b[0]), float(b[1])).to(device).eval()
    with np.load(path) as d:
        K, C = d["K"].astype(np.float32), d["C"].astype(np.float32)
    Kn = EV.normalize_K(K, stats, True, 1e-6)
    extra = EV.make_condition_maps(Path(path).parent.name, Kn.shape[0], Kn.shape[1], meta, cfg["condition_params"],
                                   cfg["transport_conditioning_stats"])
    pred = EV.stitch_predict_allT_multioutput(model, Kn, 320, 160, device, extra_channels=extra)
    gt = EV.gt_log10(np.clip(C, 0, None), 1e-12)
    return K, gt, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master_root", required=True)
    ap.add_argument("--pool_root", required=True)
    ap.add_argument("--registry_v7", required=True)
    ap.add_argument("--fresh_registry", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--metadata_ext", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--setting_mean", required=True)
    ap.add_argument("--master_per_case", required=True, help="per_case_metrics.csv of the same checkpoint, original test")
    ap.add_argument("--pool_per_case", required=True, help="per_case_metrics.csv of the same checkpoint, Pool T")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    reg = {r["param_folder"]: r for r in csv.DictReader(open(a.registry_v7))}
    test = json.loads(Path(a.split).read_text())["splits"]["test"]["param_folders"]
    geo = lambda r: (float(r["sigma2Y"]), float(r["correlation_length"]), float(r["anisotropy"]))
    cells = sorted({geo(reg[f]) for f in test})
    cell = cells[0]
    mf = [f for f in test if geo(reg[f]) == cell and float(reg[f]["long_dispersivity"]) == 6.2 and float(reg[f]["trans_ratio"]) == 0.1][0]
    fresh = [r for r in csv.DictReader(open(a.fresh_registry)) if r["pool"] == "T"]
    pf = [f"param_{r['param_id']}" for r in fresh if (float(r["sigma2Y"]), float(r["lambda"]), float(r["anisotropy"])) == cell
          and float(r["alphaL"]) == 6.2 and float(r["alphaT_over_alphaL"]) == 0.1][0]
    meta = load_transport_metadata(a.metadata_ext)
    stats = EV.load_stats(a.stats)
    ckpt = torch.load(a.ckpt, map_location=a.device)
    means = dict(np.load(a.setting_mean))
    sm = means["6.2_0.1"][DAY_IDX]
    panels = []
    for label, path in (("original test field (twin in training)", f"{a.master_root}/{mf}/real_001.npz"),
                        ("independent Pool T field", f"{a.pool_root}/{pf}/real_001.npz")):
        K, gt, pr = predict(ckpt, stats, meta, path, a.device)
        s_pr = EV.ssim_fixed(gt[DAY_IDX], pr[DAY_IDX], 14.0)
        s_sm = EV.ssim_fixed(gt[DAY_IDX], sm, 14.0)
        panels.append((label, np.log(K), gt[DAY_IDX], pr[DAY_IDX], s_pr, s_sm, path))
    fig, ax = plt.subplots(2, 4, figsize=(11, 6.2), constrained_layout=True)
    kmin, kmax = shared_conductivity_limits([p[1] for p in panels])
    for i, (label, lk, gt, pr, s_pr, s_sm, _) in enumerate(panels):
        imk = ax[i, 0].imshow(np.flipud(lk), cmap="viridis", vmin=kmin, vmax=kmax)
        for j, (img, title) in enumerate(((gt, "simulator"), (pr, f"surrogate (global SSIM {s_pr:.2f})"),
                                          (sm, f"setting mean, no $K$ ({s_sm:.2f})"))):
            imc = ax[i, j + 1].imshow(np.flipud(np.clip(img, VMIN, VMAX)), cmap="magma", vmin=VMIN, vmax=VMAX)
            if j > 0:
                ax[i, j + 1].contour(np.flipud(gt), levels=[OUTLINE], colors="cyan", linewidths=0.9, linestyles="solid")
            ax[i, j + 1].set_title(title, fontsize=9)
        ax[i, 0].set_title(f"$\\ln K$: {label}", fontsize=9)
        for x in ax[i]:
            x.set_xticks([]); x.set_yticks([])
    fig.colorbar(imk, ax=ax[:, 0], shrink=0.6, label="$\\ln K$")
    fig.colorbar(imc, ax=ax[:, 1:], shrink=0.6, label="$\\log_{10}((C+10^{-12})/C_0)$, day 365; display [-8, 0]")
    fig.suptitle("Day-365 global SSIM: one case and checkpoint; not aggregate plume SSIM", fontsize=10)
    fig.savefig(a.out, bbox_inches="tight")
    prov = {"cell_true_geology": cell, "master_case": f"{mf}/real_001", "pool_case": f"{pf}/real_001",
            "checkpoint": Path(a.ckpt).name, "checkpoint_sha256": sha(a.ckpt), "stats_sha256": sha(a.stats),
            "case_file_sha256": {p[0]: sha(p[6]) for p in panels}, "day": 365, "display_range_log10": [VMIN, VMAX],
            "outline_log10": OUTLINE, "domain": "full 600x400, no crop", "orientation": "vertical mirror of stored array",
            "conductivity_display_range_shared_lnK": [kmin, kmax],
            "displayed_metric": "day-365 global SSIM, single case/checkpoint",
            "ranking_metric": "all-time plume SSIM; not the displayed metric",
            "ssim_day365_global": {p[0]: {"surrogate": p[4], "setting_mean": p[5]} for p in panels},
            "case_rank_plume_ssim": {"original": percentile(a.master_per_case, mf, "real_001"),
                                     "pool_T": percentile(a.pool_per_case, pf, "real_001")}}
    Path(a.out).with_suffix(".json").write_text(json.dumps(prov, indent=1))
    print("cell", cell, mf, pf, [(p[0], round(p[4], 3), round(p[5], 3)) for p in panels])


def shared_conductivity_limits(fields) -> tuple[float, float]:
    """One finite numerical scale for all conductivity panels and their colorbar."""
    arrays = [np.asarray(field) for field in fields]
    if not arrays or any(not np.isfinite(field).all() or field.size == 0 for field in arrays):
        raise ValueError("Conductivity panels must be nonempty and finite")
    return min(float(field.min()) for field in arrays), max(float(field.max()) for field in arrays)


if __name__ == "__main__":
    main()
