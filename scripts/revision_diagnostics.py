"""Descriptive revision checks from released metadata and case scores, never model execution."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("global_ssim", "plume_ssim")


def flag(series: pd.Series) -> pd.Series:
    if not series.astype(str).str.lower().isin(["true", "false"]).all():
        raise ValueError("Invalid Boolean metadata")
    return series.astype(str).str.lower().eq("true")


def cost_per_decision(counts: np.ndarray, missed_cost: float) -> float:
    """Counts order TP, FP, TN, FN; FP cost fixed at one, no calibration assumed."""
    counts = np.asarray(counts, dtype=float)
    if counts.shape != (4,) or not np.isfinite(counts).all() or (counts < 0).any() or counts.sum() <= 0:
        raise ValueError("Invalid confusion counts")
    if not np.isfinite(missed_cost) or missed_cost < 0:
        raise ValueError("Invalid missed-exceedance cost")
    return float((counts[1] + missed_cost * counts[3]) / counts.sum())


def counterfactual(frame: pd.DataFrame, reference: dict) -> dict:
    keys = ["seed", "param_folder", "realization", "supplied"]
    if frame[keys].duplicated().any() or not np.isfinite(frame[list(METRICS)]).all().all():
        raise ValueError("Duplicate or nonfinite sweep scores")
    order = list(reference["matrix"][METRICS[0]])
    if set(frame.supplied) != set(order) or set(frame.true_code) != set(order):
        raise ValueError("Incomplete code coverage")
    coverage = frame.groupby(["seed", "param_folder", "realization"]).supplied.nunique()
    if not coverage.eq(6).all() or frame.seed.nunique() != 10 or frame.cell_id.nunique() != 6:
        raise ValueError("Incomplete seed/cell/case coverage")
    if not frame.groupby("seed").size().eq(174 * 6).all():
        raise ValueError("Expected 174 cases per seed, six supplied codes")
    out = {"n_seeds": 10, "n_cells": 6, "n_cases_per_seed": 174, "matrix": {}, "consistency": {}}
    for metric in METRICS:
        mat = frame.pivot_table(index="true_code", columns="supplied", values=metric, aggfunc="mean").loc[order, order]
        expected = np.array([[reference["matrix"][metric][r][c] for c in order] for r in order])
        if not np.allclose(mat.to_numpy(), expected, rtol=0, atol=1e-12):
            raise ValueError("Sweep differs from certified matrix: " + metric)
        margins = []
        for unit in ("cell_id", "seed"):
            table = frame.groupby([unit, "true_code", "supplied"])[metric].mean().unstack("supplied")[order]
            margins.append(np.array([row[t] - row.drop(t).max() for (_, t), row in table.iterrows()]))
        mc, ms = margins
        stats = {"cell_rows_correct_highest": int((mc > 0).sum()), "cell_rows_total": len(mc),
                 "seed_rows_correct_highest": int((ms > 0).sum()), "seed_rows_total": len(ms)}
        for key, value in stats.items():
            if value != reference["consistency"][metric][key]:
                raise ValueError("Sweep consistency differs: " + key)
        out["matrix"][metric] = {r: {c: float(mat.loc[r, c]) for c in order} for r in order}
        out["consistency"][metric] = stats
    return out


def qa_tables() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    summary, strata, correlations = {}, [], []
    for pool in ("T", "T2"):
        runs = pd.read_csv(ROOT / f"metadata/fresh_seed/pool_{pool}_runs.csv")
        if len(runs) != 810 or runs[["param_id", "real"]].duplicated().any():
            raise ValueError("Incomplete QA design")
        good = np.ones(len(runs), dtype=bool)
        for column in ("complete", "no_blowup", "size_ok", "k_consistent"):
            good &= flag(runs[column]).to_numpy()
        if pool == "T":
            excluded = pd.read_csv(ROOT / "metadata/fresh_seed/pool_T_qa_exclusions.csv")
            good = ~runs.apply(lambda r: (int(r.param_id), f"real_{int(r.real):03d}") in
                              set(zip(excluded.param_id, excluded.realization)), axis=1).to_numpy()
        else:
            inclusion = pd.read_csv(ROOT / "metadata/fresh_seed/pool_T2_inclusion.csv")
            merged = runs.merge(inclusion[["param_id", "real", "d8_primary"]], on=["param_id", "real"], validate="one_to_one")
            if not merged[["param_id", "real"]].equals(runs[["param_id", "real"]]):
                raise ValueError("QA inclusion order changed")
            if not np.array_equal(good, flag(merged.d8_primary)):
                raise ValueError("T2 inclusion differs from QA flags")
        runs["retained"] = good
        pure_blowup = (~flag(runs.no_blowup) & flag(runs.complete) & flag(runs.size_ok) & flag(runs.k_consistent))
        groups = runs.groupby(["cell_id", "real"]).retained.sum()
        summary[pool] = {"expected_cases": 810, "retained_cases": int(good.sum()), "excluded_cases": int((~good).sum()),
                         "pure_no_blowup_failures": int(pure_blowup.sum()),
                         "other_run_or_size_failures": int((~good).sum() - pure_blowup.sum()),
                         "cell_realization_groups_with_retained_cases": int((groups > 0).sum()),
                         "retained_codes_per_group": {str(k): int(v) for k, v in groups.value_counts().sort_index().items()}}
        expected_count = 794 if pool == "T" else 797
        if int(good.sum()) != expected_count:
            raise ValueError("QA retained count differs")
        for (cell, transport), block in runs.groupby(["cell_id", "transport_id"]):
            strata.append({"pool": pool, "cell_id": int(cell), "transport_id": int(transport),
                           "expected": len(block), "retained": int(block.retained.sum()),
                           "excluded": int((~block.retained).sum()),
                           "solver_overshoot_only": int(pure_blowup.loc[block.index].sum()),
                           "other_run_or_size": int((~block.retained).sum() - pure_blowup.loc[block.index].sum())})
        for (cell, real), block in runs.groupby(["cell_id", "real"]):
            values = block.max_abs_r_master.dropna().to_numpy(dtype=float)
            if not len(values) or not np.isfinite(values).all():
                raise ValueError("Missing field-screen summary")
            correlations.append({"pool": pool, "cell_id": int(cell), "realization": int(real),
                                 "max_abs_r_master": float(values.max()), "qa_rows_with_correlation": len(values)})
    baseline = pd.read_csv(ROOT / "results/v7/per_case/setting_mean/setting_mean_A_poolT_per_case.csv")
    reg = pd.read_csv(ROOT / "metadata/fresh_seed/param_registry_fresh_seed.csv")
    reg = reg[reg.pool == "T"].assign(param_folder=lambda f: f.param_id.map(lambda p: f"param_{p}"))
    joined = baseline.merge(reg[["param_folder", "cell_id"]], on="param_folder", validate="many_to_one")
    if len(joined) != 794 or joined.groupby(["cell_id", "realization"]).ngroups != 135:
        raise ValueError("Per-case group count differs from registry")
    return summary, pd.DataFrame(strata), pd.DataFrame(correlations)


def build(output: Path) -> None:
    if output.exists():
        raise FileExistsError("Use a new output directory")
    sweep_path = ROOT / "results/v9/counterfactual_case_scores.csv"
    reference_path = ROOT / "figures/v7/fig_v7_counterfactual.json"
    sweep = counterfactual(pd.read_csv(sweep_path), json.loads(reference_path.read_text()))
    counts, strata, correlations = qa_tables()
    d = json.loads((ROOT / "results/v8/review_diagnostics.json").read_text())
    rows, costs = [], []
    for model, record in d["wells"].items():
        total = np.zeros(4)
        for well, row in record["per_well"].items():
            confusion = np.array(row["confusion_pooled_seed_repeats"])
            tp, fp, tn, fn = confusion
            total += confusion
            n = record["n_seeds"]
            if confusion.sum() != 794 * n:
                raise ValueError("Well confusion denominator differs")
            rows.append({"model": model, "well": well, "n_seeds": n, "cases_per_seed": 794,
                         "positive_cases": int((tp + fn) / n), "negative_cases": int((tn + fp) / n),
                         "tp_seed_pooled": int(tp), "fp_seed_pooled": int(fp), "tn_seed_pooled": int(tn), "fn_seed_pooled": int(fn),
                         "fnr": row["false_negative_rate"]["estimate"], "fnr_lo": row["false_negative_rate"]["ci95"][0],
                         "fnr_hi": row["false_negative_rate"]["ci95"][1],
                         "fpr": row["false_positive_rate"]["estimate"], "fpr_lo": row["false_positive_rate"]["ci95"][0],
                         "fpr_hi": row["false_positive_rate"]["ci95"][1]})
        for ratio in (1., 5., 10., 20.):
            costs.append({"model": model, "false_negative_cost": ratio, "false_positive_cost": 1.,
                          "descriptive_cost_per_case_well_decision": cost_per_decision(total, ratio)})
    output.mkdir(parents=True)
    strata.to_csv(output / "qa_strata.csv", index=False)
    correlations.to_csv(output / "field_screen_summary.csv", index=False)
    wells = pd.DataFrame(rows)
    for _, block in wells.groupby("well"):
        if block.positive_cases.nunique() != 1 or block.negative_cases.nunique() != 1:
            raise ValueError("Predictors do not share true well outcomes")
    wells.to_csv(output / "per_well_confusion.csv", index=False)
    pd.DataFrame(costs).to_csv(output / "decision_cost_sensitivity.csv", index=False)
    grouped_path = ROOT / "results/v7/analysis/v7_grouped_results.json"
    grouped = json.loads(grouped_path.read_text())["grouped"]
    summary = {"scope": "Post-review descriptive reconstruction; no new model scores or field verification",
               "qa": counts, "counterfactual": sweep,
               "well_cost_assumption": "FP cost=1; FN costs=1,5,10,20 are illustrative choices, not measured deployment costs; no threshold tuning or cost-optimal claim",
               "source_hashes": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                 [sweep_path, reference_path, grouped_path, ROOT / "results/v8/review_diagnostics.json",
                                  *sorted((ROOT / "metadata/fresh_seed").glob("pool_T*csv")),
                                  ROOT / "metadata/fresh_seed/param_registry_fresh_seed.csv"]}}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    # Compact appendix material. Cell-level strata and full per-well counts remain in the CSVs.
    lines = ["% Generated descriptive appendix from permitted metadata; no primary numeric macros changed.",
             r"\section{Quality exclusions and monitoring-well support}\label{app:review}",
             r"\begin{table}[!ht]\centering\small",
             r"\caption{Case-level quality exclusions by transport setting. Every setting starts with 135 cases; reasons overlap. Detailed cell-level counts are released.}\label{tab:qa}",
             r"\begin{tabular}{lrrrr}\toprule Setting & T excluded & T retained & T2 excluded & T2 retained\\\midrule"]
    pairs = [(0.62, .1), (.62, 1.), (6.2, .1), (6.2, 1.), (62., .1), (62., 1.)]
    for i, pair in enumerate(pairs):
        vals = [strata[(strata.pool == p) & (strata.transport_id == i)] for p in ("T", "T2")]
        lines.append(f"$({pair[0]:g},{pair[1]:g})$ & {vals[0].excluded.sum()} & {vals[0].retained.sum()} & {vals[1].excluded.sum()} & {vals[1].retained.sum()} " + r"\\")
    lines += [r"\bottomrule\end{tabular}\end{table}",
              r"\begin{table}[!ht]\centering\small",
              r"\caption{Monitoring-well support and missed-exceedance rates on Pool T. P/N are positive/negative simulation cases, not seed repetitions. Rates pool 10 learned-model seeds or one deterministic reference; released CSVs give confusion counts, false-positive rates and conditional stream/seed intervals. These seven training-selected probes are not a deployment network.}\label{tab:wells}",
              r"\begin{tabular}{lrrr}\toprule Offset (m); P/N & Log-setting mean & Matched & $K$ only\\\midrule"]
    for well in sorted(wells.well.unique(), key=lambda w: tuple(int(x) for x in w[1:].split('_'))):
        block = wells[wells.well == well].set_index("model")
        ref = block.loc["log_setting_mean"]
        label = well.removeprefix("w").replace("_", ",")
        lines.append(f"$({label})$; {ref.positive_cases:.0f}/{ref.negative_cases:.0f} & {ref.fnr:.3f} & {block.loc['matched'].fnr:.3f} & {block.loc['k_only'].fnr:.3f} " + r"\\")
    lines += [r"\bottomrule\end{tabular}\end{table}"]
    lines += [r"\begin{table}[!ht]\centering\small",
              r"\caption{Grouped-split SSIM sensitivity to target quality screening, 10-seed means. The all-case set includes four flagged test realizations (174 cases); the screened set omits them (170). This is not performance on the missing/corrupt fresh-pool cases.}\label{tab:screened}",
              r"\begin{tabular}{lrrrr}\toprule & \multicolumn{2}{c}{Global SSIM} & \multicolumn{2}{c}{Plume SSIM}\\",
              r"Predictor & All & Screened & All & Screened\\\midrule"]
    keys = {"cta_ffl_matched": "CTA-UNet, matched", "cta_ffl_k_only": "CTA-UNet, $K$ only",
            "cta_ffl_permuted": "CTA-UNet, cyclic", "cta_ffl_constant_placebo": "CTA-UNet, constant",
            "cta_ffl_independent_nuisance": "CTA-UNet, nuisance", "ms_tmo_matched": "MS-TMO, matched"}
    for key, label in keys.items():
        vals = [grouped[source][key][metric]['mean'] for metric in METRICS
                for source in ('summary', 'summary_ssim_screened')]
        lines.append(label + ' & ' + ' & '.join(f'{v:.4f}' for v in vals) + r"\\")
    lines += [r"\bottomrule\end{tabular}\end{table}"]
    (output / "revision_tables.tex").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "PASS", "qa": counts, "counterfactual_rows": len(pd.read_csv(sweep_path)), "well_rows": len(wells)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
