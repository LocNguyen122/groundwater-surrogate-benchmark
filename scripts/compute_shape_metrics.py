"""Recompute the six-family EAAI plume-shape panel from prediction caches.

Metric definitions and the empty-mask convention match the manuscript. Input,
cache, and output roots are explicit command-line arguments so the script does
not depend on author-machine paths.
"""
from __future__ import annotations
import csv
import argparse
import glob
import json
import math
import os
import time
from typing import Dict, List, Tuple

import numpy as np

from src.shared.eval.centroid_error import concentration_weighted_centroid


PROJECT = os.environ.get("TRANSPORT_PROJECT_ROOT", os.getcwd())
PRED_ROOT = os.path.join(PROJECT, "predictions_eaai")
DATA_ROOT = os.path.join(PROJECT, "data", "T25_TSTEP_OVERRIDE_FINAL_FLIPPED")
OUT_DIR = os.path.join(PROJECT, "results", "tables")

EPS_C = 1e-12
PLUME_THRESH = 1e-8
MIN_PLUME_PIX = 64
T = 25
H, W = 600, 400

# Family spec: list of (family_label, [(source_dir_label, source_dir, seeds)])
def default_families(pred_root: str) -> List[Tuple[str, List[Tuple[str, str, List[int]]]]]:
    """Return the six-family cache layout used by the manuscript shape panel."""
    return [
        ("CTA-UNet+FFL TC", [("canonical", os.path.join(pred_root, "cta_unet_ffl_transport_conditioned"), list(range(10)))]),
        ("MS-TMO-UNet TC", [
            ("canonical", os.path.join(pred_root, "ms_tmo_transport_conditioned"), list(range(5))),
            ("recovered", os.path.join(pred_root, "ms_tmo_transport_conditioned_recovery"), list(range(5, 10))),
        ]),
        ("CNN/U-Net TC", [("canonical", os.path.join(pred_root, "cnn_transport_conditioned"), list(range(10)))]),
        ("Pix2Pix TC", [("canonical", os.path.join(pred_root, "pix2pix_transport_conditioned"), list(range(10)))]),
        ("FNO TC", [("canonical", os.path.join(pred_root, "fno_transport_conditioned"), list(range(10)))]),
        ("CTA-UNet+FFL K-only", [("canonical", os.path.join(pred_root, "cta_unet_ffl_konly"), list(range(1, 10)))]),
    ]


FAMS = default_families(PRED_ROOT)


def load_families_json(path: str) -> List[Tuple[str, List[Tuple[str, str, List[int]]]]]:
    """Load a custom family layout.

    Expected schema:
    [
      {
        "family": "CTA-UNet+FFL TC",
        "sources": [
          {"source": "canonical", "dir": "/path/to/family", "seeds": [0, 1]}
        ]
      }
    ]
    """
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    families: List[Tuple[str, List[Tuple[str, str, List[int]]]]] = []
    for item in payload:
        label = item["family"]
        sources = []
        for source in item.get("sources", []):
            sources.append((source.get("source", "canonical"), source["dir"], [int(s) for s in source["seeds"]]))
        families.append((label, sources))
    return families


def process_seed(family_label: str, source_dir_label: str, seed: int, seed_dir: str) -> Dict[str, object]:
    npzs = sorted(glob.glob(os.path.join(seed_dir, "*.pred.npz")))
    assert len(npzs) == 170, f"expected 170 NPZ in {seed_dir}, got {len(npzs)}"

    n_pairs = 0; n_both_empty = 0; n_false_alarm = 0; n_miss = 0; n_miss_eligible = 0
    iou_sum = 0.0; dice_sum = 0.0; n_iou = 0
    fp_area_frac_sum = 0.0; fn_area_frac_sum = 0.0
    pred_area_frac_sum = 0.0; gt_area_frac_sum = 0.0
    abs_plume_area_rel_err_sum = 0.0; n_area_rel = 0
    centroid_err_sum = 0.0; n_centroid = 0
    mass_signed_sum = 0.0; mass_abs_sum = 0.0

    iou_pt = np.zeros(T, dtype=np.float64); n_iou_pt = np.zeros(T, dtype=np.int64)
    dice_pt = np.zeros(T, dtype=np.float64)
    fp_pt = np.zeros(T, dtype=np.float64); fn_pt = np.zeros(T, dtype=np.float64)
    centroid_pt = np.zeros(T, dtype=np.float64); n_centroid_pt = np.zeros(T, dtype=np.int64)
    mass_signed_pt = np.zeros(T, dtype=np.float64); mass_abs_pt = np.zeros(T, dtype=np.float64)
    n_pt = np.zeros(T, dtype=np.int64); miss_pt = np.zeros(T, dtype=np.int64)
    miss_eligible_pt = np.zeros(T, dtype=np.int64)

    HW = float(H * W)

    for p in npzs:
        z = np.load(p)
        pred_log = z["pred_log"].astype(np.float32)
        src = str(z["source_file"])
        stem = os.path.basename(p).replace(".pred.npz", "")
        parent, _ = stem.split("__", 1)
        C = np.load(os.path.join(DATA_ROOT, parent, src))["C"].astype(np.float32)
        for t in range(T):
            gt_phys = np.clip(C[t], 0.0, None)
            pr_phys = np.clip(np.power(10.0, pred_log[t]) - EPS_C, 0.0, None)
            gt_mask = gt_phys > PLUME_THRESH
            pr_mask = pr_phys > PLUME_THRESH
            gt_sum = int(gt_mask.sum()); pr_sum = int(pr_mask.sum())
            n_pairs += 1; n_pt[t] += 1
            inter = int(np.logical_and(gt_mask, pr_mask).sum())
            union = gt_sum + pr_sum - inter
            fp = int(np.logical_and(np.logical_not(gt_mask), pr_mask).sum())
            fn = int(np.logical_and(gt_mask, np.logical_not(pr_mask)).sum())
            fp_area_frac_sum += fp/HW; fn_area_frac_sum += fn/HW
            pred_area_frac_sum += pr_sum/HW; gt_area_frac_sum += gt_sum/HW
            fp_pt[t] += fp/HW; fn_pt[t] += fn/HW
            mass_signed = float(pr_phys.sum() - gt_phys.sum())
            mass_signed_sum += mass_signed; mass_abs_sum += abs(mass_signed)
            mass_signed_pt[t] += mass_signed; mass_abs_pt[t] += abs(mass_signed)
            if gt_sum > 0:
                abs_plume_area_rel_err_sum += abs(pr_sum-gt_sum)/gt_sum; n_area_rel += 1
            if gt_sum >= MIN_PLUME_PIX:
                n_miss_eligible += 1; miss_eligible_pt[t] += 1
            cg = concentration_weighted_centroid(gt_phys)
            cp = concentration_weighted_centroid(pr_phys)
            if cg is not None and cp is not None:
                d = math.hypot(cg[0]-cp[0], cg[1]-cp[1])
                centroid_err_sum += d; n_centroid += 1
                centroid_pt[t] += d; n_centroid_pt[t] += 1
            if gt_sum == 0 and pr_sum == 0:
                n_both_empty += 1; continue
            if gt_sum == 0 and pr_sum > 0:
                n_false_alarm += 1
                iou_sum += 0.0; dice_sum += 0.0; n_iou += 1
                iou_pt[t] += 0.0; dice_pt[t] += 0.0; n_iou_pt[t] += 1
                continue
            if gt_sum > 0 and pr_sum == 0:
                if gt_sum >= MIN_PLUME_PIX:
                    n_miss += 1; miss_pt[t] += 1
                iou_sum += 0.0; dice_sum += 0.0; n_iou += 1
                iou_pt[t] += 0.0; dice_pt[t] += 0.0; n_iou_pt[t] += 1
                continue
            iou = inter/union if union > 0 else 0.0
            dice = (2*inter)/(gt_sum+pr_sum) if (gt_sum+pr_sum) > 0 else 0.0
            iou_sum += iou; dice_sum += dice; n_iou += 1
            iou_pt[t] += iou; dice_pt[t] += dice; n_iou_pt[t] += 1

    safe = lambda num, den: float(num/den) if den > 0 else float("nan")
    metrics = {
        "family": family_label, "source": source_dir_label, "seed": seed,
        "n_pairs": n_pairs, "n_both_empty": n_both_empty,
        "n_false_alarm": n_false_alarm, "n_miss": n_miss,
        "n_miss_eligible": n_miss_eligible,
        "iou_mean": safe(iou_sum, n_iou),
        "dice_mean": safe(dice_sum, n_iou),
        "fp_area_frac_mean": safe(fp_area_frac_sum, n_pairs),
        "fn_area_frac_mean": safe(fn_area_frac_sum, n_pairs),
        "pred_area_frac_mean": safe(pred_area_frac_sum, n_pairs),
        "gt_area_frac_mean": safe(gt_area_frac_sum, n_pairs),
        "abs_plume_area_rel_err_mean": safe(abs_plume_area_rel_err_sum, n_area_rel),
        "centroid_err_mean_px": safe(centroid_err_sum, n_centroid),
        "plume_miss_rate": safe(n_miss, n_miss_eligible),
        "mass_signed_mean": safe(mass_signed_sum, n_pairs),
        "mass_abs_mean": safe(mass_abs_sum, n_pairs),
    }
    by_t = []
    for t in range(T):
        n = int(n_pt[t]); n_i = int(n_iou_pt[t]); n_c = int(n_centroid_pt[t])
        by_t.append({
            "family": family_label, "source": source_dir_label, "seed": seed, "timestep": t,
            "n_pairs": n, "n_iou_pairs": n_i, "n_centroid_pairs": n_c,
            "iou_mean": safe(iou_pt[t], n_i),
            "dice_mean": safe(dice_pt[t], n_i),
            "fp_area_frac_mean": safe(fp_pt[t], n),
            "fn_area_frac_mean": safe(fn_pt[t], n),
            "centroid_err_mean_px": safe(centroid_pt[t], n_c),
            "mass_signed_mean": safe(mass_signed_pt[t], n),
            "mass_abs_mean": safe(mass_abs_pt[t], n),
            "miss_count": int(miss_pt[t]),
            "miss_eligible_count": int(miss_eligible_pt[t]),
            "plume_miss_rate": safe(miss_pt[t], miss_eligible_pt[t]),
        })
    return {"metrics": metrics, "by_timestep": by_t}


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    global DATA_ROOT, OUT_DIR, FAMS, PLUME_THRESH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DATA_ROOT, help="Directory containing parameter folders and NPZ simulation files.")
    parser.add_argument("--pred-root", default=PRED_ROOT, help="Directory containing per-family prediction-cache folders.")
    parser.add_argument("--out-dir", default=OUT_DIR, help="Output directory for the three shape-metric CSV files.")
    parser.add_argument("--families-json", default=None, help="Optional custom family/source/seed JSON specification.")
    parser.add_argument("--tau", type=float, default=PLUME_THRESH, help="Plume threshold in physical concentration units.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned family/source/seed layout and exit.")
    args = parser.parse_args()

    DATA_ROOT = os.path.abspath(args.data_root)
    OUT_DIR = os.path.abspath(args.out_dir)
    PLUME_THRESH = float(args.tau)
    FAMS = load_families_json(args.families_json) if args.families_json else default_families(os.path.abspath(args.pred_root))

    if args.dry_run:
        plan = []
        for label, source_groups in FAMS:
            for source_label, srcdir, seeds in source_groups:
                present = [
                    s
                    for s in seeds
                    if os.path.isdir(os.path.join(srcdir, f"seed{s}"))
                    and glob.glob(os.path.join(srcdir, f"seed{s}", "*.pred.npz"))
                ]
                plan.append({"family": label, "source": source_label, "dir": srcdir, "requested_seeds": seeds, "present_seeds": present})
        print(json.dumps({"status": "dry-run", "tau": PLUME_THRESH, "data_root": DATA_ROOT, "out_dir": OUT_DIR, "families": plan}, indent=2))
        return

    per_seed_rows: List[Dict[str, object]] = []
    by_t_rows: List[Dict[str, object]] = []
    for label, source_groups in FAMS:
        for source_label, srcdir, seeds in source_groups:
            present = [s for s in seeds
                       if os.path.isdir(os.path.join(srcdir, f"seed{s}"))
                       and glob.glob(os.path.join(srcdir, f"seed{s}", "*.pred.npz"))]
            print(f"[{label}] source={source_label} dir={srcdir} present_seeds={present}", flush=True)
            for s in present:
                t0 = time.time()
                seed_dir = os.path.join(srcdir, f"seed{s}")
                r = process_seed(label, source_label, s, seed_dir)
                m = r["metrics"]
                print(f"  {label}/{source_label}/seed{s} done {time.time()-t0:.1f}s iou={m['iou_mean']:.4f} dice={m['dice_mean']:.4f} centroid_px={m['centroid_err_mean_px']:.3f} miss_rate={m['plume_miss_rate']:.4f}", flush=True)
                per_seed_rows.append(m)
                by_t_rows.extend(r["by_timestep"])

    per_seed_fields = [
        "family", "source", "seed", "n_pairs", "n_both_empty", "n_false_alarm", "n_miss", "n_miss_eligible",
        "iou_mean", "dice_mean", "fp_area_frac_mean", "fn_area_frac_mean",
        "pred_area_frac_mean", "gt_area_frac_mean", "abs_plume_area_rel_err_mean",
        "centroid_err_mean_px", "plume_miss_rate",
        "mass_signed_mean", "mass_abs_mean",
    ]
    per_seed_csv = os.path.join(OUT_DIR, "shape_metrics_per_seed_2026-06-11.csv")
    write_csv(per_seed_csv, per_seed_rows, per_seed_fields)
    print(f"Wrote {per_seed_csv}", flush=True)

    by_t_fields = [
        "family", "source", "seed", "timestep", "n_pairs", "n_iou_pairs", "n_centroid_pairs",
        "iou_mean", "dice_mean", "fp_area_frac_mean", "fn_area_frac_mean",
        "centroid_err_mean_px", "mass_signed_mean", "mass_abs_mean", "miss_count",
        "miss_eligible_count", "plume_miss_rate",
    ]
    by_t_csv = os.path.join(OUT_DIR, "shape_metrics_by_timestep_2026-06-11.csv")
    write_csv(by_t_csv, by_t_rows, by_t_fields)
    print(f"Wrote {by_t_csv}", flush=True)

    # Per-family summary
    metric_keys = [
        "iou_mean", "dice_mean", "fp_area_frac_mean", "fn_area_frac_mean",
        "pred_area_frac_mean", "gt_area_frac_mean", "abs_plume_area_rel_err_mean",
        "centroid_err_mean_px", "plume_miss_rate",
        "mass_signed_mean", "mass_abs_mean",
    ]
    by_fam: Dict[str, List[Dict[str, object]]] = {}
    for r in per_seed_rows:
        by_fam.setdefault(r["family"], []).append(r)
    summary_rows: List[Dict[str, object]] = []
    for fam, rows in by_fam.items():
        seeds_used = sorted(int(r["seed"]) for r in rows)
        sources = sorted({r["source"] for r in rows})
        row = {
            "family": fam,
            "seeds": ",".join(str(s) for s in seeds_used),
            "n_seeds": len(seeds_used),
            "sources": "+".join(sources),
        }
        for k in metric_keys:
            vals = [float(r[k]) for r in rows if r[k] is not None and not (isinstance(r[k], float) and math.isnan(r[k]))]
            if not vals:
                row[f"{k}_mean"] = float("nan"); row[f"{k}_std"] = float("nan"); continue
            m = float(np.mean(vals)); sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            row[f"{k}_mean"] = m; row[f"{k}_std"] = sd
        summary_rows.append(row)
    summary_fields = ["family", "seeds", "n_seeds", "sources"] + [f"{k}_{stat}" for k in metric_keys for stat in ("mean", "std")]
    summary_csv = os.path.join(OUT_DIR, "shape_metrics_summary_2026-06-11.csv")
    write_csv(summary_csv, summary_rows, summary_fields)
    print(f"Wrote {summary_csv}", flush=True)


if __name__ == "__main__":
    main()
