"""Compute threshold sensitivity for focal transport surrogate benchmark EAAI plume-shape metrics.

This is a zero-training, CPU-only analysis over existing prediction caches.
It evaluates whether the main plume-support conclusions are stable when the
physical plume threshold is changed from the manuscript value 1e-8 to nearby
values 1e-9 and 1e-7.

The prediction and data roots are explicit command-line arguments. Outputs are
written to a caller-selected directory and never overwrite existing files.
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


DATA_ROOT = ""

EPS_C = 1e-12
THRESHOLDS = [1e-9, 1e-8, 1e-7]
MIN_PLUME_PIX = 64
T = 25
H, W = 600, 400
HW = float(H * W)

def default_families(pred_root: Path) -> List[Tuple[str, List[Tuple[str, str, List[int]]]]]:
    return [
        ("CTA-UNet+FFL TC", [("canonical", str(pred_root / "cta_unet_ffl_transport_conditioned"), list(range(10)))]),
        ("MS-TMO-UNet TC", [
            ("canonical", str(pred_root / "ms_tmo_transport_conditioned"), list(range(5))),
            ("recovered", str(pred_root / "ms_tmo_transport_conditioned_recovery"), list(range(5, 10))),
        ]),
        ("CTA-UNet+FFL K-only", [("canonical", str(pred_root / "cta_unet_ffl_konly"), list(range(1, 10)))]),
    ]


def safe(num: float, den: float) -> float:
    return float(num / den) if den > 0 else float("nan")


def mean_std(vals: Iterable[float]) -> Tuple[float, float]:
    arr = np.array([v for v in vals if not math.isnan(float(v))], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def process_seed(family: str, source: str, seed: int, seed_dir: str) -> List[Dict[str, object]]:
    files = sorted(glob.glob(os.path.join(seed_dir, "*.pred.npz")))
    if len(files) != 170:
        raise RuntimeError(f"expected 170 prediction files under {seed_dir}, got {len(files)}")

    stats = {
        tau: {
            "n_pairs": 0,
            "n_iou": 0,
            "n_both_empty": 0,
            "n_false_alarm": 0,
            "n_miss": 0,
            "iou_sum": 0.0,
            "dice_sum": 0.0,
            "fp_frac_sum": 0.0,
            "fn_frac_sum": 0.0,
            "pred_area_frac_sum": 0.0,
            "gt_area_frac_sum": 0.0,
            "area_rel_sum": 0.0,
            "n_area_rel": 0,
        }
        for tau in THRESHOLDS
    }

    for pred_path in files:
        with np.load(pred_path) as z:
            pred_log = z["pred_log"].astype(np.float32)
            src = str(z["source_file"])
        stem = os.path.basename(pred_path).replace(".pred.npz", "")
        parent, _ = stem.split("__", 1)
        gt_path = os.path.join(DATA_ROOT, parent, src)
        C = np.load(gt_path)["C"].astype(np.float32)
        pred_phys = np.clip(np.power(10.0, pred_log) - EPS_C, 0.0, None)
        gt_phys = np.clip(C, 0.0, None)

        for tau in THRESHOLDS:
            s = stats[tau]
            gt_mask_all = gt_phys > tau
            pr_mask_all = pred_phys > tau
            for t in range(T):
                gt_mask = gt_mask_all[t]
                pr_mask = pr_mask_all[t]
                gt_sum = int(gt_mask.sum())
                pr_sum = int(pr_mask.sum())
                inter = int(np.logical_and(gt_mask, pr_mask).sum())
                union = gt_sum + pr_sum - inter
                fp = int(np.logical_and(~gt_mask, pr_mask).sum())
                fn = int(np.logical_and(gt_mask, ~pr_mask).sum())

                s["n_pairs"] += 1
                s["fp_frac_sum"] += fp / HW
                s["fn_frac_sum"] += fn / HW
                s["pred_area_frac_sum"] += pr_sum / HW
                s["gt_area_frac_sum"] += gt_sum / HW

                if gt_sum > 0:
                    s["area_rel_sum"] += abs(pr_sum - gt_sum) / gt_sum
                    s["n_area_rel"] += 1

                if gt_sum == 0 and pr_sum == 0:
                    s["n_both_empty"] += 1
                    continue
                if gt_sum == 0 and pr_sum > 0:
                    s["n_false_alarm"] += 1
                    s["n_iou"] += 1
                    continue
                if gt_sum > 0 and pr_sum == 0:
                    if gt_sum >= MIN_PLUME_PIX:
                        s["n_miss"] += 1
                    s["n_iou"] += 1
                    continue

                iou = inter / union if union > 0 else 0.0
                dice = (2 * inter) / (gt_sum + pr_sum) if (gt_sum + pr_sum) > 0 else 0.0
                s["iou_sum"] += iou
                s["dice_sum"] += dice
                s["n_iou"] += 1

    rows = []
    for tau in THRESHOLDS:
        s = stats[tau]
        rows.append(
            {
                "family": family,
                "source": source,
                "seed": seed,
                "threshold": tau,
                "n_pairs": s["n_pairs"],
                "n_iou_pairs": s["n_iou"],
                "n_both_empty": s["n_both_empty"],
                "n_false_alarm": s["n_false_alarm"],
                "n_miss": s["n_miss"],
                "iou_mean": safe(s["iou_sum"], s["n_iou"]),
                "dice_mean": safe(s["dice_sum"], s["n_iou"]),
                "fp_area_frac_mean": safe(s["fp_frac_sum"], s["n_pairs"]),
                "fn_area_frac_mean": safe(s["fn_frac_sum"], s["n_pairs"]),
                "pred_area_frac_mean": safe(s["pred_area_frac_sum"], s["n_pairs"]),
                "gt_area_frac_mean": safe(s["gt_area_frac_sum"], s["n_pairs"]),
                "abs_plume_area_rel_err_mean": safe(s["area_rel_sum"], s["n_area_rel"]),
                "plume_miss_rate": safe(s["n_miss"], s["n_pairs"]),
            }
        )
    return rows


def write_csv(path: str, rows: List[Dict[str, object]], fields: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    global DATA_ROOT
    DATA_ROOT = str(args.data_root)
    families = default_families(args.pred_root)
    per_seed_path = str(args.out_dir / "threshold_sensitivity_per_seed.csv")
    summary_path = str(args.out_dir / "threshold_sensitivity_summary.csv")
    report_path = str(args.out_dir / "threshold_sensitivity_report.md")
    for p in [per_seed_path, summary_path, report_path]:
        if os.path.exists(p):
            raise SystemExit(f"Refusing to overwrite existing output: {p}")

    per_seed_rows: List[Dict[str, object]] = []
    for family, groups in families:
        for source, srcdir, seeds in groups:
            for seed in seeds:
                seed_dir = os.path.join(srcdir, f"seed{seed}")
                if not os.path.isdir(seed_dir):
                    print(f"[skip] missing {family} {source} seed{seed}: {seed_dir}", flush=True)
                    continue
                t0 = time.time()
                rows = process_seed(family, source, seed, seed_dir)
                per_seed_rows.extend(rows)
                msg = ", ".join(
                    f"tau={r['threshold']:.0e} IoU={float(r['iou_mean']):.4f} Dice={float(r['dice_mean']):.4f}"
                    for r in rows
                )
                print(f"[done] {family} {source} seed{seed} in {time.time()-t0:.1f}s: {msg}", flush=True)

    fields = [
        "family",
        "source",
        "seed",
        "threshold",
        "n_pairs",
        "n_iou_pairs",
        "n_both_empty",
        "n_false_alarm",
        "n_miss",
        "iou_mean",
        "dice_mean",
        "fp_area_frac_mean",
        "fn_area_frac_mean",
        "pred_area_frac_mean",
        "gt_area_frac_mean",
        "abs_plume_area_rel_err_mean",
        "plume_miss_rate",
    ]
    write_csv(per_seed_path, per_seed_rows, fields)

    by_key: Dict[Tuple[str, float], List[Dict[str, object]]] = defaultdict(list)
    for row in per_seed_rows:
        by_key[(str(row["family"]), float(row["threshold"]))].append(row)

    metric_keys = [
        "iou_mean",
        "dice_mean",
        "fp_area_frac_mean",
        "fn_area_frac_mean",
        "abs_plume_area_rel_err_mean",
        "plume_miss_rate",
    ]
    summary_rows: List[Dict[str, object]] = []
    for (family, tau), rows in sorted(by_key.items()):
        seeds = sorted(int(r["seed"]) for r in rows)
        out: Dict[str, object] = {
            "family": family,
            "threshold": tau,
            "n_seeds": len(seeds),
            "seeds": ",".join(str(s) for s in seeds),
        }
        for key in metric_keys:
            m, s = mean_std(float(r[key]) for r in rows)
            out[f"{key}_mean"] = m
            out[f"{key}_std"] = s
        summary_rows.append(out)

    summary_fields = ["family", "threshold", "n_seeds", "seeds"] + [
        f"{key}_{stat}" for key in metric_keys for stat in ("mean", "std")
    ]
    write_csv(summary_path, summary_rows, summary_fields)

    def fmt(m: float, s: float, nd: int = 4) -> str:
        return f"{m:.{nd}f} +/- {s:.{nd}f}"

    lines = [
        "# transport surrogate benchmark EAAI Threshold Sensitivity",
        "",
        "Scope: zero-training, CPU-only sensitivity analysis from existing prediction caches.",
        "Thresholds: `1e-9`, `1e-8`, and `1e-7` physical concentration.",
        "Families: CTA-UNet+FFL TC, MS-TMO-UNet TC, and CTA-UNet+FFL K-only.",
        "",
        "## Summary",
        "",
        "| family | threshold | seeds | IoU | Dice | FP area frac | FN area frac | area rel.err. |",
        "|" + "|".join(["-" * 3, "-" * 3 + ":", "-" * 3 + ":", "-" * 3 + ":", "-" * 3 + ":", "-" * 3 + ":", "-" * 3 + ":", "-" * 3 + ":"]) + "|",
    ]
    for row in summary_rows:
        lines.append(
            "| {family} | {threshold:.0e} | {n_seeds} | {iou} | {dice} | {fp} | {fn} | {area} |".format(
                family=row["family"],
                threshold=float(row["threshold"]),
                n_seeds=int(row["n_seeds"]),
                iou=fmt(float(row["iou_mean_mean"]), float(row["iou_mean_std"]), 4),
                dice=fmt(float(row["dice_mean_mean"]), float(row["dice_mean_std"]), 4),
                fp=fmt(float(row["fp_area_frac_mean_mean"]), float(row["fp_area_frac_mean_std"]), 5),
                fn=fmt(float(row["fn_area_frac_mean_mean"]), float(row["fn_area_frac_mean_std"]), 5),
                area=fmt(float(row["abs_plume_area_rel_err_mean_mean"]), float(row["abs_plume_area_rel_err_mean_std"]), 4),
            )
        )
    lines += [
        "",
        "## Interpretation Guardrail",
        "",
        "Use this analysis as threshold robustness evidence only. It does not change the primary manuscript threshold,",
        "does not add new training, and does not imply field deployment without ground-truth evaluation data.",
        "",
        "## Output Files",
        "",
        f"- `{per_seed_path}`",
        f"- `{summary_path}`",
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {per_seed_path}", flush=True)
    print(f"Wrote {summary_path}", flush=True)
    print(f"Wrote {report_path}", flush=True)
    print("THRESHOLD_SENSITIVITY_DONE", flush=True)


if __name__ == "__main__":
    main()
