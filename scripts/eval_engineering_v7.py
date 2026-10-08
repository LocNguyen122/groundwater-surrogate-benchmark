"""Engineering metrics and the no-conductivity setting-mean predictor (pre-registration O5).

Predictions are produced exactly as the unified evaluator produces them (same model loader, bounded output,
standardization, constant code maps and Hann-blended stitching are imported from it); global and plume SSIM are
recomputed with its functions as a cross-check. Per case it adds, against the simulator (C0 = 1, grid 5 m,
model input orientation: source at stored row 200, column 199, plume moving toward larger row index):
  iou_1e-3, iou_1e-4      : mean over days of IoU of {C >= tau}, days where either mask is non-empty
  area_rel_err_1e-3_d365  : |area_pred - area_true| / area_true of {C >= 1e-3} on day 365
  centroid_err_m_d365     : distance (m) between concentration-weighted centroids of {C >= 1e-4} on day 365
  exceed_w<dn>_<lat>_{true,pred}: does C at a monitoring well (dn m down-gradient, lat m lateral) reach 1e-3
                          by day 365; seven wells chosen from training-data exceedance rates (deviation D5)
Modes: --ckpt <best.pt> (surrogate) or --setting_mean <npz> (mean bounded log field per transport setting,
built from training files by --build_setting_mean).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from src.shared.data.concentration import LOG10_MAX, LOG10_MIN
from src.shared.data.io import filter_excluded_files, flatten_param_folders, load_param_split
from src.shared.data.transport_conditioning import load_transport_metadata, param_folder_from_path
from src.shared.eval import eval_transport_unified_logc_global_plumemask as EV

DX_M = 5.0
SOURCE = (200, 199)
# (down-gradient m, lateral m) monitoring wells; chosen from training-data exceedance rates (deviation D5)
WELLS_M = ((500, 400), (500, -400), (1000, 400), (1000, -400), (1500, 600), (1500, -600), (1900, 0))
TAUS = (1e-3, 1e-4)
EXCEED_TAU = 1e-3


def centroid(c: np.ndarray, tau: float):
    m = c >= tau
    if not m.any():
        return None
    ys, xs = np.nonzero(m)
    w = c[m]
    return np.array([np.average(ys, weights=w), np.average(xs, weights=w)])


def engineering(gt: np.ndarray, pr: np.ndarray) -> dict:
    """gt, pr: physical concentration (T, H, W)."""
    out = {}
    for tau in TAUS:
        ious = []
        for t in range(gt.shape[0]):
            a, b = gt[t] >= tau, pr[t] >= tau
            u = np.logical_or(a, b).sum()
            if u:
                ious.append(np.logical_and(a, b).sum() / u)
        out[f"iou_{tau:g}"] = float(np.mean(ious)) if ious else float("nan")
    at, ap = (gt[-1] >= 1e-3).sum(), (pr[-1] >= 1e-3).sum()
    out["area_rel_err_1e-3_d365"] = float(abs(ap - at) / at) if at else float("nan")
    ct, cp = centroid(gt[-1], 1e-4), centroid(pr[-1], 1e-4)
    out["centroid_err_m_d365"] = float(np.linalg.norm(ct - cp) * DX_M) if ct is not None and cp is not None else float("nan")
    for dn, lat in WELLS_M:
        row, col = SOURCE[0] + int(dn / DX_M), SOURCE[1] + int(lat / DX_M)
        tag = f"w{dn}_{lat:+d}"
        out[f"exceed_{tag}_true"] = int(gt[:, row, col].max() >= EXCEED_TAU)
        out[f"exceed_{tag}_pred"] = int(pr[:, row, col].max() >= EXCEED_TAU)
    return out


def ssim_pair(gt_phys: np.ndarray, pr_log: np.ndarray, eps_c: float, args) -> tuple[float, float]:
    g, p = [], []
    for t in range(gt_phys.shape[0]):
        gp = np.clip(gt_phys[t], 0.0, None)
        gl = EV.gt_log10(gp, eps_c)
        g.append(EV.ssim_fixed(gl, pr_log[t], data_range=14.0, bounds=(LOG10_MIN, LOG10_MAX)))
        mask = gp > args.plume_thresh
        if int(mask.sum()) >= args.plume_min_pixels:
            bb = EV.bbox_from_mask(mask, pad=args.plume_pad)
            if bb is not None and (bb[1] - bb[0]) >= 7 and (bb[3] - bb[2]) >= 7:
                y0, y1, x0, x1 = bb
                p.append(EV.ssim_fixed(gl[y0:y1, x0:x1], pr_log[t][y0:y1, x0:x1], data_range=14.0,
                                       bounds=(LOG10_MIN, LOG10_MAX)))
    return float(np.mean(g)), (float(np.mean(p)) if p else float("nan"))


def build_setting_mean(args) -> None:
    train_p, _, _ = load_param_split(args.split_json)
    files = filter_excluded_files(flatten_param_folders(args.data_root, train_p), args.exclusion_csv)
    meta = load_transport_metadata(args.transport_metadata_csv)
    acc, cnt = {}, {}
    for f in files:
        m = meta[param_folder_from_path(f)]
        key = f"{m['alpha_L']:g}_{m['alpha_T_ratio']:g}"
        with np.load(f) as d:
            y = np.clip(EV.gt_log10(np.clip(d["C"].astype(np.float64), 0, None), 1e-12), LOG10_MIN, LOG10_MAX)
        acc[key] = acc.get(key, 0.0) + y
        cnt[key] = cnt.get(key, 0) + 1
    np.savez_compressed(args.build_setting_mean, **{k: (acc[k] / cnt[k]).astype(np.float32) for k in acc})
    Path(args.build_setting_mean + ".json").write_text(json.dumps({"n_files_per_setting": cnt, "split": args.split_json}))
    print("setting means:", cnt)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--split_json", required=True)
    ap.add_argument("--exclusion_csv", default="")
    ap.add_argument("--transport_metadata_csv", required=True)
    ap.add_argument("--stats_json", default="")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--setting_mean", default="")
    ap.add_argument("--build_setting_mean", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--plume_thresh", type=float, default=1e-8)
    ap.add_argument("--plume_pad", type=int, default=8)
    ap.add_argument("--plume_min_pixels", type=int, default=64)
    ap.add_argument("--max_files", type=int, default=0)
    ap.add_argument("--out_csv", default="")
    args = ap.parse_args()
    if args.build_setting_mean:
        return build_setting_mean(args)

    _, _, test_p = load_param_split(args.split_json)
    files = filter_excluded_files(flatten_param_folders(args.data_root, test_p), args.exclusion_csv)
    if args.max_files:
        files = files[: args.max_files]
    meta = load_transport_metadata(args.transport_metadata_csv)
    model = None
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location=args.device)
        cfg = ckpt.get("cfg", {})
        stats = EV.load_stats(args.stats_json)
        model, _, mode = EV.build_model_from_ckpt(ckpt, args.device)
        assert mode == "multioutput"
        if cfg.get("output_parameterization") == "bounded_sigmoid_log10":
            b = cfg.get("output_bounds", [LOG10_MIN, LOG10_MAX])
            model = EV._BoundedLog10Output(model, float(b[0]), float(b[1])).to(args.device).eval()
        cmeta = load_transport_metadata(args.transport_metadata_csv) if cfg.get("use_transport_conditioning") else None
    else:
        means = dict(np.load(args.setting_mean))
    rows = []
    for f in files:
        with np.load(f) as d:
            K, C = d["K"].astype(np.float32), d["C"].astype(np.float32)
        pf = param_folder_from_path(f)
        if model is not None:
            Kn = EV.normalize_K(K, stats, bool(cfg.get("use_logK", True)), float(cfg.get("eps_k", 1e-6)))
            extra = (EV.make_condition_maps(pf, Kn.shape[0], Kn.shape[1], cmeta, cfg.get("condition_params", []),
                                            cfg.get("transport_conditioning_stats", {})) if cmeta is not None else None)
            pr_log = EV.stitch_predict_allT_multioutput(model, Kn, 320, 160, args.device, extra_channels=extra)
        else:
            m = meta[pf]
            pr_log = means[f"{m['alpha_L']:g}_{m['alpha_T_ratio']:g}"]
        g_ssim, p_ssim = ssim_pair(C, pr_log, 1e-12, args)
        row = {"param_folder": pf, "realization": Path(f).stem, "global_ssim": g_ssim, "plume_ssim": p_ssim}
        row.update(engineering(np.clip(C, 0, None), EV.log10_to_phys(pr_log, 1e-12)))
        rows.append(row)
    with open(args.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} cases to {args.out_csv}")


if __name__ == "__main__":
    main()
