"""Exploratory: does a surrogate carry conductivity-specific information beyond the setting mean?

For each evaluation case, with m the setting mean of the requested transport setting (bounded log10 field), y the
simulator field and p the surrogate prediction (both bounded log10), the anomaly correlation is
    corr(p - m, y - m)
over all 25 output times inside the bounding box of the true plume support (C > 1e-8 at any time, padded by 8 cells).
A value near 1 means the surrogate reproduces the case-specific departure from the mean plume; a value near 0 means
its departure from the mean is unrelated to the true one. Not pre-registered; reported as exploratory.

Writes a per-case CSV (param_folder, realization, anomaly_corr, plume_ssim_mean_vs_true for reference is not computed).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_v7_example_figure import predict  # noqa: E402  (same prediction path as Figure 3)

from src.shared.data.concentration import LOG10_MAX, LOG10_MIN  # noqa: E402
from src.shared.data.io import filter_excluded_files, flatten_param_folders, load_param_split  # noqa: E402
from src.shared.data.transport_conditioning import load_transport_metadata  # noqa: E402
from src.shared.eval import eval_transport_unified_logc_global_plumemask as EV  # noqa: E402


def setting_key(meta: dict, folder: str) -> str:
    r = meta[folder]
    return f"{float(r['alpha_L']):g}_{float(r['alpha_T_ratio']):g}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--split_json", required=True)
    ap.add_argument("--split_part", default="test")
    ap.add_argument("--exclusion_csv", default="")
    ap.add_argument("--metadata_csv", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--setting_mean", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    tr, va, te = load_param_split(a.split_json)
    params = {"train": tr, "val": va, "test": te}[a.split_part]
    files = filter_excluded_files(flatten_param_folders(a.data_root, params), a.exclusion_csv)
    meta_rows = {r["param_folder"]: r for r in csv.DictReader(open(a.metadata_csv, encoding="utf-8-sig"))}
    meta = load_transport_metadata(a.metadata_csv)
    stats = EV.load_stats(a.stats)
    ckpt = torch.load(a.ckpt, map_location=a.device)
    means = dict(np.load(a.setting_mean))
    rows = []
    for i, path in enumerate(files, 1):
        folder = Path(path).parent.name
        _, gt, pr = predict(ckpt, stats, meta, path, a.device)
        y = np.clip(gt, LOG10_MIN, LOG10_MAX)
        p = np.clip(pr, LOG10_MIN, LOG10_MAX)
        m = means[setting_key(meta_rows, folder)]
        support = (y > np.log10(1e-8 + 1e-12)).any(axis=0)
        box = EV.bbox_from_mask(support, pad=8)
        if box is None:
            continue
        r0, r1, c0, c1 = box
        dy = (y - m)[:, r0:r1, c0:c1].ravel()
        dp = (p - m)[:, r0:r1, c0:c1].ravel()
        ac = float(np.corrcoef(dp, dy)[0, 1]) if dy.std() > 0 and dp.std() > 0 else float("nan")
        rows.append({"param_folder": folder, "realization": Path(path).stem, "anomaly_corr": ac})
        if i % 100 == 0:
            print(f"{i}/{len(files)}", flush=True)
    with open(a.out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["param_folder", "realization", "anomaly_corr"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    v = np.array([r["anomaly_corr"] for r in rows], dtype=float)
    print(f"cases {len(v)} median {np.nanmedian(v):.3f} mean {np.nanmean(v):.3f} frac>0 {(v > 0).mean():.3f}")


if __name__ == "__main__":
    main()
