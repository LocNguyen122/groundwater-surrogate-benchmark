"""Obj1 EAAI v7: every inferential number of the manuscript, from per-case result files.

Two evaluation populations:
  master  : the grouped-split test cells of the original corpus (6 cells, each test field has an
            amplitude-rescaled training twin) -- reported as a sensitivity analysis;
  poolT   : the fresh-seed independent pool (27 cells, 5 independent draws per cell) -- primary, analysed
            as pre-registered in fresh_seed_campaign_2026-10/prereg/PREREGISTRATION_OBJ1_V7_2026-10-03.md.
Cells are keyed by TRUE geology (sigma2Y, correlation length, anisotropy) from the simulation records
(metadata/parameter_registry_162_v7.csv; deviation D4). Inference: analysis/cell_stats.py.

Usage:
  python analyze_v7.py --master_root <v7_data/master_test_per_case> [--poolT_root <fetched Pool T tree>] \
      --registry_v7 <repo_v7/metadata/parameter_registry_162_v7.csv> --fresh_registry <design/param_registry_fresh_seed.csv> \
      --split <repo_v7/splits/param_split_grouped_k_v5.json> --solver_qa <repo_v7/metadata/solver_qa_exclusions_v7.csv> \
      [--setting_mean_csv <per-case CSV of the setting-mean predictor on Pool T>] --out v7_results.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cell_stats import decide, summarize

PRIMARY_PAIR = (6.2, 1.0)
LOCO = {"loco_0p62_0p1": (0.62, 0.1), "loco_0p62_1p0": (0.62, 1.0), "loco_62p0_0p1": (62.0, 0.1),
        "loco_62p0_1p0": (62.0, 1.0), "loco_6p2_0p1": (6.2, 0.1), "compositional": PRIMARY_PAIR}
CONF = "runs_v5_confirmatory_20260715"
MU = "runs_method_upgrade_pilot_20260723"
TOPUP = "runs_v6_factorization_topup_20260921"
V4R = "runs_v4_joint_continuous_20260920"
RD = "runs_v6_realization_disjoint_20260921"


class Data:
    def __init__(self, args):
        reg7 = list(csv.DictReader(open(args.registry_v7, encoding="utf-8")))
        self.cell = {r["param_folder"]: (float(r["sigma2Y"]), float(r["correlation_length"]), float(r["anisotropy"]))
                     for r in reg7}
        self.pair = {r["param_folder"]: (float(r["long_dispersivity"]), float(r["trans_ratio"])) for r in reg7}
        self.cluster = {}                       # Pool T folder -> randf seed class (deviation D6)
        if args.fresh_registry:
            classes = json.loads(Path(args.seed_classes).read_text())["class_of_seed"] if args.seed_classes else {}
            for r in csv.DictReader(open(args.fresh_registry, encoding="utf-8")):
                pf = f"param_{int(r['param_id'])}"
                self.cell[pf] = (float(r["sigma2Y"]), float(r["lambda"]), float(r["anisotropy"]))
                self.pair[pf] = (float(r["alphaL"]), float(r["alphaT_over_alphaL"]))
                if r["pool"] == "T":
                    self.cluster[pf] = f"class_{classes[str(int(r['seed']))]}" if classes else f"cell_{r['cell_id']}"
        sp = json.loads(Path(args.split).read_text())["splits"]
        self.test_cells = {self.cell[f] for f in sp["test"]["param_folders"]}
        self.train_cells = {self.cell[f] for f in sp["train"]["param_folders"]}
        self.flagged = {(r["param_folder"], r["realization"]) for r in csv.DictReader(open(args.solver_qa))}
        self.master_root = Path(args.master_root)
        self.pool_root = Path(args.poolT_root) if args.poolT_root else None
        self.setting_mean_csv = args.setting_mean_csv

    def runs(self, rel: str, pop: str, ckpt: str = "best") -> dict[int, pd.DataFrame]:
        """seed -> per-case frame for one condition directory (relative to the run roots)."""
        out = {}
        base = (self.master_root if pop == "master" else self.pool_root) / rel
        for sd in sorted(base.glob("seed*")):
            f = sd / "per_case_metrics.csv" if pop == "master" else sd / "e1_fresh_seed_pool_T" / ckpt / "per_case_metrics.csv"
            if f.is_file():
                df = pd.read_csv(f)
                df["geology"] = df["param_folder"].map(self.cell)
                # inference cluster: geological cell on the master, randf seed class on Pool T (D6)
                df["cell"] = df["geology"] if pop == "master" else df["param_folder"].map(self.cluster)
                df["pair"] = df["param_folder"].map(self.pair)
                df["flagged"] = [(a, b) in self.flagged for a, b in zip(df["param_folder"], df["realization"])]
                out[int(sd.name[4:])] = df
        return out


def paired(left: dict, right: dict, metric: str, pairs=None, cells=None, screen=False, sign=1.0) -> pd.DataFrame:
    """Long table (seed, cell, unit, d) of seed-paired, case-paired differences, averaged over the transport
    settings in `pairs` within each (cell, realization) unit."""
    rows = []
    for s in sorted(set(left) & set(right)):
        m = left[s].merge(right[s][["param_folder", "realization", metric]], on=["param_folder", "realization"],
                          suffixes=("_l", "_r"))
        if pairs is not None:
            m = m[m["pair"].isin(pairs)]
        if cells is not None:
            m = m[m["geology"].isin(cells)]
        if screen:
            m = m[~m["flagged"]]
        m = m.assign(d=sign * (m[f"{metric}_l"] - m[f"{metric}_r"])).dropna(subset=["d"])
        g = m.groupby(["cell", "realization"])["d"].mean().reset_index()
        g["seed"] = s
        rows.append(g.rename(columns={"realization": "unit"}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["seed", "cell", "unit", "d"])


def describe(runs: dict, metric: str, pairs=None, cells=None, screen=False) -> dict:
    """Seed-level mean and sample SD of the case mean (the 'mean +- SD across seeds' of the tables)."""
    vals = []
    for s, df in sorted(runs.items()):
        m = df
        if pairs is not None:
            m = m[m["pair"].isin(pairs)]
        if cells is not None:
            m = m[m["geology"].isin(cells)]
        if screen:
            m = m[~m["flagged"]]
        vals.append(float(m[metric].mean()))
    v = np.array(vals)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "n_seeds": len(v)}


ALL_PAIRS = [(a, r) for a in (0.62, 6.2, 62.0) for r in (0.1, 1.0)]


def contrast(left, right, metric, rule="superiority", margin=0.0, **kw):
    res = summarize(paired(left, right, metric, **kw))
    res["decision"] = decide(res, rule, margin)
    return res


def grouped_section(D: Data) -> dict:
    out = {}
    conds = {c: D.runs(f"{CONF}/{c}", "master") for c in
             ("cta_ffl_matched", "cta_ffl_k_only", "cta_ffl_permuted", "cta_ffl_constant_placebo",
              "cta_ffl_independent_nuisance", "ms_tmo_matched")}
    out["summary"] = {c: {m: describe(r, m, screen=m in ("ice_per_timestep", "concentration_mae")) for m in
                          ("global_ssim", "plume_ssim", "ice_per_timestep", "concentration_mae")}
                      for c, r in conds.items()}
    out["summary_ssim_screened"] = {c: {m: describe(r, m, screen=True) for m in ("global_ssim", "plume_ssim")}
                                    for c, r in conds.items()}
    eff = {}
    for other in ("cta_ffl_k_only", "cta_ffl_permuted", "cta_ffl_constant_placebo", "cta_ffl_independent_nuisance"):
        eff[other] = {m: contrast(conds["cta_ffl_matched"], conds[other], m) for m in ("global_ssim", "plume_ssim")}
        for m in ("ice_per_timestep", "concentration_mae"):   # lower is better: direction-adjusted
            eff[other][m] = contrast(conds["cta_ffl_matched"], conds[other], m, screen=True, sign=-1.0)
    eff["cta_vs_mstmo"] = {m: contrast(conds["cta_ffl_matched"], conds["ms_tmo_matched"], m) for m in ("global_ssim", "plume_ssim")}
    for m in ("ice_per_timestep", "concentration_mae"):
        eff["cta_vs_mstmo"][m] = contrast(conds["cta_ffl_matched"], conds["ms_tmo_matched"], m, screen=True, sign=-1.0)
    out["effects"] = eff

    comp = {}
    for split, pair in LOCO.items():
        v1, v0 = D.runs(f"{MU}/{split}/v1_factorized_levels", "master"), D.runs(f"{MU}/{split}/v0_raw_channels", "master")
        comp[split] = {"pair": pair, "v1": describe(v1, "plume_ssim", pairs=[pair]), "v0": describe(v0, "plume_ssim", pairs=[pair]),
                       "v1_minus_v0": contrast(v1, v0, "plume_ssim", pairs=[pair])}
        v2 = D.runs(f"{MU}/{split}/v2_joint_embedding", "master")
        if v2:
            comp[split]["v2"] = describe(v2, "plume_ssim", pairs=[pair])
    out["compositional"] = comp
    std = {v: D.runs(f"{MU}/grouped_standard/{v}", "master") for v in
           ("v0_raw_channels", "v1_factorized_levels", "v2_joint_embedding", "v3_factorized_continuous")}
    out["in_design"] = {v: describe(r, "plume_ssim") for v, r in std.items()}
    out["in_design"]["v1_minus_v0"] = contrast(std["v1_factorized_levels"], std["v0_raw_channels"], "plume_ssim",
                                               rule="noninferiority", margin=0.01)
    v3 = {**D.runs(f"{MU}/compositional/v3_factorized_continuous", "master"), **D.runs(f"{TOPUP}/compositional/v3_factorized_continuous", "master")}
    v4 = {**D.runs(f"{V4R}/compositional/v4_joint_continuous", "master"), **D.runs(f"{TOPUP}/compositional/v4_joint_continuous", "master")}
    v0 = D.runs(f"{MU}/compositional/v0_raw_channels", "master")
    v1 = D.runs(f"{MU}/compositional/v1_factorized_levels", "master")
    out["decomposition"] = {"n_seeds_v3": len(v3), "n_seeds_v4": len(v4),
                            "v3_minus_v4": contrast(v3, v4, "plume_ssim", pairs=[PRIMARY_PAIR]),
                            "v4_minus_v0": contrast(v4, v0, "plume_ssim", pairs=[PRIMARY_PAIR]),
                            "v1_minus_v4": contrast(v1, v4, "plume_ssim", pairs=[PRIMARY_PAIR]),
                            "v3": describe(v3, "plume_ssim", pairs=[PRIMARY_PAIR]),
                            "v4": describe(v4, "plume_ssim", pairs=[PRIMARY_PAIR])}
    ms1 = D.runs("runs_mstmo_crossarch_full_20260807/compositional", "master")
    ms0 = D.runs("runs_mstmo_raw_loco_full_20260810/compositional", "master")
    out["mstmo"] = {"v1": describe(ms1, "plume_ssim", pairs=[PRIMARY_PAIR]), "v0": describe(ms0, "plume_ssim", pairs=[PRIMARY_PAIR]),
                    "v1_minus_v0": contrast(ms1, ms0, "plume_ssim", pairs=[PRIMARY_PAIR])}
    return out


def pool_section(D: Data, ckpt: str) -> dict:
    out = {}
    P = lambda rel: D.runs(rel, "poolT", ckpt)
    trained = [p for p in ALL_PAIRS if p != PRIMARY_PAIR]
    conds = {c: P(f"{CONF}/{c}") for c in ("cta_ffl_matched", "cta_ffl_k_only", "cta_ffl_permuted",
                                            "cta_ffl_constant_placebo", "cta_ffl_independent_nuisance", "ms_tmo_matched")}
    out["summary"] = {c: {m: describe(r, m) for m in ("global_ssim", "plume_ssim")} for c, r in conds.items() if r}
    # O1 twin inflation: same checkpoints, same 6 held-out cells; master test fields (twins) minus Pool T fields
    mm, pm = D.runs(f"{CONF}/cta_ffl_matched", "master"), conds["cta_ffl_matched"]
    o1 = {}
    for name, (mr, pr) in {"matched": (mm, pm), "k_only": (D.runs(f"{CONF}/cta_ffl_k_only", "master"), conds["cta_ffl_k_only"])}.items():
        rows = []
        for s in sorted(set(mr) & set(pr)):
            a = mr[s][mr[s]["geology"].isin(D.test_cells)].groupby(["geology", "realization"])["plume_ssim"].mean().reset_index()
            b = pr[s][pr[s]["geology"].isin(D.test_cells)]
            clus = b.groupby("geology")["cell"].first()      # Pool T seed class of each held-out cell (D6)
            b = b.groupby(["geology", "realization"])["plume_ssim"].mean().reset_index()
            for c in sorted(D.test_cells):
                av, bv = a[a["geology"] == c]["plume_ssim"].to_numpy(), b[b["geology"] == c]["plume_ssim"].to_numpy()
                for q in range(min(len(av), len(bv))):          # arbitrary pairing of independent random draws
                    rows.append({"seed": s, "cell": clus[c], "unit": q, "d": av[q] - bv[q]})
        res = summarize(pd.DataFrame(rows))
        res["decision"] = decide(res)
        res["twin_mean"] = describe(mr, "plume_ssim", cells=D.test_cells)
        res["pool_mean"] = describe(pr, "plume_ssim", cells=D.test_cells)
        o1[name] = res
    out["O1_twin_inflation"] = o1
    # O2 codes on independent geology
    o2 = {"matched_minus_k_only": contrast(conds["cta_ffl_matched"], conds["cta_ffl_k_only"], "plume_ssim"),
          "matched_minus_k_only_global": contrast(conds["cta_ffl_matched"], conds["cta_ffl_k_only"], "global_ssim"),
          "matched_minus_cyclic": contrast(conds["cta_ffl_matched"], conds["cta_ffl_permuted"], "plume_ssim",
                                           rule="equivalence", margin=0.01),
          "matched_minus_constant": contrast(conds["cta_ffl_matched"], conds["cta_ffl_constant_placebo"], "plume_ssim"),
          "matched_minus_nuisance": contrast(conds["cta_ffl_matched"], conds["cta_ffl_independent_nuisance"], "plume_ssim"),
          "heldout_cells_matched_minus_k_only": contrast(conds["cta_ffl_matched"], conds["cta_ffl_k_only"], "plume_ssim",
                                                         cells=D.test_cells)}
    out["O2_codes"] = o2
    # O3 compositional on independent geology
    v1, v0 = P(f"{MU}/compositional/v1_factorized_levels"), P(f"{MU}/compositional/v0_raw_channels")
    o3 = {"primary_v1_minus_v0": contrast(v1, v0, "plume_ssim", rule="superiority_strict_negative", pairs=[PRIMARY_PAIR]),
          "v1": describe(v1, "plume_ssim", pairs=[PRIMARY_PAIR]), "v0": describe(v0, "plume_ssim", pairs=[PRIMARY_PAIR]),
          "in_design_v1_minus_v0": contrast(v1, v0, "plume_ssim", rule="noninferiority", margin=0.01, pairs=trained)}
    loco = {}
    for split, pair in LOCO.items():
        if split == "compositional":
            continue
        a, b = P(f"{MU}/{split}/v1_factorized_levels"), P(f"{MU}/{split}/v0_raw_channels")
        if a and b:
            loco[split] = {"pair": pair, **contrast(a, b, "plume_ssim", pairs=[pair])}
    ps = sorted((k, v["p_signflip"]) for k, v in loco.items())
    order = sorted(range(len(ps)), key=lambda i: ps[i][1])
    adj, running = {}, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i][1]))
        adj[ps[i][0]] = running
    for k in loco:
        loco[k]["p_holm"] = adj[k]
    o3["loco_secondary"] = loco
    v3 = {**P(f"{MU}/compositional/v3_factorized_continuous"), **P(f"{TOPUP}/compositional/v3_factorized_continuous")}
    v4 = {**P(f"{V4R}/compositional/v4_joint_continuous"), **P(f"{TOPUP}/compositional/v4_joint_continuous")}
    o3["v3_minus_v4"] = contrast(v3, v4, "plume_ssim", pairs=[PRIMARY_PAIR])
    o3["v4_minus_v0"] = contrast(v4, v0, "plume_ssim", pairs=[PRIMARY_PAIR])
    ms1, ms0 = P("runs_mstmo_crossarch_full_20260807/compositional"), P("runs_mstmo_raw_loco_full_20260810/compositional")
    if ms1 and ms0:
        o3["mstmo_v1_minus_v0"] = contrast(ms1, ms0, "plume_ssim", pairs=[PRIMARY_PAIR])
    out["O3_compositional"] = o3
    # O5a / O6c: no-conductivity setting mean (master training fields) as a constant "run" for every seed
    if D.setting_mean_csv:
        sm = pd.read_csv(D.setting_mean_csv)
        sm["geology"] = sm["param_folder"].map(D.cell)
        sm["cell"] = sm["param_folder"].map(D.cluster)
        sm["pair"] = sm["param_folder"].map(D.pair)
        sm["flagged"] = False
        rep_sm = lambda runs: {s: sm for s in runs}
        out["setting_mean_A"] = {m: float(sm[m].mean()) for m in ("global_ssim", "plume_ssim")}
        out["O5a_matched_minus_setting_mean"] = contrast(conds["cta_ffl_matched"], rep_sm(conds["cta_ffl_matched"]), "plume_ssim")
        out["O5a_matched_minus_setting_mean_global"] = contrast(conds["cta_ffl_matched"], rep_sm(conds["cta_ffl_matched"]), "global_ssim")
        out["O5a_k_only_minus_setting_mean"] = contrast(conds["cta_ffl_k_only"], rep_sm(conds["cta_ffl_k_only"]), "plume_ssim")
        v0c = {s: r for s, r in v0.items() if s <= 4}
        o6c = contrast(v0c, rep_sm(v0c), "plume_ssim", pairs=trained)
        o6c["decision"] = ("SURROGATE BETTER" if o6c["ci_low"] > 0 and o6c["p_signflip"] < 0.05 else
                           "BASELINE BETTER" if o6c["ci_high"] < 0 and o6c["p_signflip"] < 0.05 else "INCONCLUSIVE")
        out["O6c_A_v0_minus_A_setting_mean"] = o6c
    out["O5b_engineering"] = engineering_summary(D, ckpt)
    # E1-H1 (shared pre-registration): realization-disjoint V0 on trained pairs, all cells
    rd = {v: P(f"{RD}/compositional/{v}") for v in ("v0_raw_channels", "k_only", "v1_factorized_levels")}
    if rd["v0_raw_channels"]:
        cm = summarize(pd.DataFrame([{"seed": 0, "cell": c, "unit": q, "d": d}
                                     for c, q, d in _unit_means(rd["v0_raw_channels"], trained)]))
        S = cm["estimate"]
        f = lambda x: (x - 0.5068) / (0.9312 - 0.5068)
        out["H1_twin_fraction"] = {"S": S, "S_ci": [cm["ci_low"], cm["ci_high"]], "f": f(S),
                                   "f_ci": [f(cm["ci_low"]), f(cm["ci_high"])],
                                   "decision": ("SUPPORTED" if f(cm["ci_high"]) < 0.25 else
                                                "NOT SUPPORTED" if f(cm["ci_low"]) > 0.50 else "INCONCLUSIVE"),
                                   "codes_minus_k_only": contrast(rd["v0_raw_channels"], rd["k_only"], "plume_ssim", pairs=trained)}
    return out


def e2_section(D, ckpt: str, e2_root: str, setting_mean_b_csv: str) -> dict:
    """Pre-registered O4 (addendum 1) and O6 (addendum 2): retraining on independent geology (training set B)."""
    trained = [p for p in ALL_PAIRS if p != PRIMARY_PAIR]
    B = lambda name: _e2_runs(D, e2_root, name, ckpt)
    a_v0 = {s: r for s, r in D.runs(f"{MU}/compositional/v0_raw_channels", "poolT", ckpt).items() if s <= 4}
    b = {n: B(n) for n in ("k_only", "v0_raw_channels", "v1_factorized_levels", "v4_joint_continuous", "fno_v0_raw_channels")}
    out = {"summary": {n: {m: describe(r, m, pairs=trained) for m in ("global_ssim", "plume_ssim")} for n, r in b.items() if r},
           "summary_withheld": {n: describe(r, "plume_ssim", pairs=[PRIMARY_PAIR]) for n, r in b.items() if r}}
    out["O4a_Bv0_minus_Av0"] = contrast(b["v0_raw_channels"], a_v0, "plume_ssim", pairs=trained)
    out["O4b_Bv1_minus_Bv0_withheld"] = contrast(b["v1_factorized_levels"], b["v0_raw_channels"], "plume_ssim",
                                                 rule="superiority_strict_negative", pairs=[PRIMARY_PAIR])
    out["O4c_Bv0_minus_Bkonly"] = contrast(b["v0_raw_channels"], b["k_only"], "plume_ssim", pairs=trained)
    if b["fno_v0_raw_channels"]:
        out["O4c_fno_minus_cta_v0"] = contrast(b["fno_v0_raw_channels"], b["v0_raw_channels"], "plume_ssim", pairs=trained)
    out["Bv4_minus_Bv0_withheld"] = contrast(b["v4_joint_continuous"], b["v0_raw_channels"], "plume_ssim", pairs=[PRIMARY_PAIR])
    out["Bv1_minus_Bv4_withheld"] = contrast(b["v1_factorized_levels"], b["v4_joint_continuous"], "plume_ssim", pairs=[PRIMARY_PAIR])
    sm = pd.read_csv(setting_mean_b_csv)
    sm["geology"] = sm["param_folder"].map(D.cell); sm["cell"] = sm["param_folder"].map(D.cluster)
    sm["pair"] = sm["param_folder"].map(D.pair); sm["flagged"] = False
    out["setting_mean_B"] = {m: float(sm[sm["pair"].isin(trained)][m].mean()) for m in ("global_ssim", "plume_ssim")}
    for key, name in (("O6a_Bv0_minus_B_setting_mean", "v0_raw_channels"), ("O6b_Bkonly_minus_B_setting_mean", "k_only")):
        r = contrast(b[name], {s: sm for s in b[name]}, "plume_ssim", pairs=trained)
        r["decision"] = ("SURROGATE BETTER" if r["ci_low"] > 0 and r["p_signflip"] < 0.05 else
                         "BASELINE BETTER" if r["ci_high"] < 0 and r["p_signflip"] < 0.05 else "INCONCLUSIVE")
        out[key] = r
    return out


def t2_runs(D, rel: str, ckpt: str, t2_folders: set, t2_cluster: dict) -> dict:
    """T2 replication: carried folders read from the Pool T outputs, re-seeded folders from the T2-new outputs;
    clusters are the 27 cells (all independent in T2)."""
    out = {}
    base = D.pool_root / rel
    for sd in sorted(base.glob("seed*")):
        parts = []
        for sub in ("e1_fresh_seed_pool_T", "e1_fresh_seed_pool_T2new"):
            f = sd / sub / ckpt / "per_case_metrics.csv"
            if f.is_file():
                parts.append(pd.read_csv(f))
        if len(parts) < 2:
            continue
        df = pd.concat(parts, ignore_index=True)
        df = df[df["param_folder"].isin(t2_folders)].copy()
        df["geology"] = df["param_folder"].map(D.cell)
        df["cell"] = df["param_folder"].map(t2_cluster)
        df["pair"] = df["param_folder"].map(D.pair)
        df["flagged"] = False
        out[int(sd.name[4:])] = df
    return out


def t2_section(D, ckpt: str, t2_registry: str, t2_inclusion: str) -> dict:
    reg = [r for r in csv.DictReader(open(t2_registry)) if r["pool"] == "T2"]
    keep = {(f"param_{int(r['param_id'])}", f"real_{int(r['real']):03d}") for r in csv.DictReader(open(t2_inclusion))
            if r["d8_primary"] == "True"}
    folders = {f"param_{int(r['param_id'])}" for r in reg}
    clus = {f"param_{int(r['param_id'])}": f"cell_{r['cell_id']}" for r in reg}
    for r in reg:
        pf = f"param_{int(r['param_id'])}"
        D.cell[pf] = (float(r["sigma2Y"]), float(r["lambda"]), float(r["anisotropy"]))
        D.pair[pf] = (float(r["alphaL"]), float(r["alphaT_over_alphaL"]))
    R = lambda rel: {s: df[[(a, b) in keep for a, b in zip(df["param_folder"], df["realization"])]]
                     for s, df in t2_runs(D, rel, ckpt, folders, clus).items()}
    trained = [p for p in ALL_PAIRS if p != PRIMARY_PAIR]
    m, k = R(f"{CONF}/cta_ffl_matched"), R(f"{CONF}/cta_ffl_k_only")
    v1, v0 = R(f"{MU}/compositional/v1_factorized_levels"), R(f"{MU}/compositional/v0_raw_channels")
    rd = R(f"{RD}/compositional/v0_raw_channels")
    out = {"summary": {n: describe(r, "plume_ssim") for n, r in (("matched", m), ("k_only", k), ("rd_v0", rd)) if r},
           "codes": contrast(m, k, "plume_ssim"), "codes_global": contrast(m, k, "global_ssim"),
           "comp_primary": contrast(v1, v0, "plume_ssim", rule="superiority_strict_negative", pairs=[PRIMARY_PAIR])}
    out["n_cases_per_seed"] = int(len(next(iter(m.values()))))
    if D.setting_mean_csv and getattr(D, "setting_mean_t2new_csv", ""):
        sm = pd.concat([pd.read_csv(D.setting_mean_csv), pd.read_csv(D.setting_mean_t2new_csv)], ignore_index=True)
        sm = sm[sm["param_folder"].isin(folders) &
                [(a, b) in keep for a, b in zip(sm["param_folder"], sm["realization"])]].copy()
        sm["geology"] = sm["param_folder"].map(D.cell)
        sm["cell"] = sm["param_folder"].map(clus)
        sm["pair"] = sm["param_folder"].map(D.pair)
        sm["flagged"] = False
        out["setting_mean"] = {k2: float(sm[k2].mean()) for k2 in ("global_ssim", "plume_ssim")}
        out["matched_minus_setting_mean"] = contrast(m, {s: sm for s in m}, "plume_ssim")
        out["rd_v0_minus_setting_mean"] = contrast(rd, {s: sm for s in rd}, "plume_ssim", pairs=trained)
    return out


def e2_t2_section(D, ckpt: str, e2_root: str, t2_registry: str, t2_inclusion: str, sm_b_t: str, sm_b_t2new: str) -> dict:
    """Pre-specified T2 replication of O6a/O6b and O4b for the B-trained models (27 independent cells)."""
    reg = [r for r in csv.DictReader(open(t2_registry)) if r["pool"] == "T2"]
    keep = {(f"param_{int(r['param_id'])}", f"real_{int(r['real']):03d}") for r in csv.DictReader(open(t2_inclusion))
            if r["d8_primary"] == "True"}
    folders = {f"param_{int(r['param_id'])}" for r in reg}
    clus = {f"param_{int(r['param_id'])}": f"cell_{r['cell_id']}" for r in reg}
    for r in reg:
        pf = f"param_{int(r['param_id'])}"
        D.cell[pf] = (float(r["sigma2Y"]), float(r["lambda"]), float(r["anisotropy"]))
        D.pair[pf] = (float(r["alphaL"]), float(r["alphaT_over_alphaL"]))
    saved = D.pool_root
    D.pool_root = Path(e2_root)
    try:
        R = lambda name: {s: df[[(a, b) in keep for a, b in zip(df["param_folder"], df["realization"])]]
                          for s, df in t2_runs(D, f"compositional/{name}", ckpt, folders, clus).items()}
        b = {n: R(n) for n in ("k_only", "v0_raw_channels", "v1_factorized_levels", "v4_joint_continuous")}
    finally:
        D.pool_root = saved
    trained = [p for p in ALL_PAIRS if p != PRIMARY_PAIR]
    sm = pd.concat([pd.read_csv(sm_b_t), pd.read_csv(sm_b_t2new)], ignore_index=True)
    sm = sm[sm["param_folder"].isin(folders) & [(a, c) in keep for a, c in zip(sm["param_folder"], sm["realization"])]].copy()
    sm["geology"] = sm["param_folder"].map(D.cell); sm["cell"] = sm["param_folder"].map(clus)
    sm["pair"] = sm["param_folder"].map(D.pair); sm["flagged"] = False
    out = {"summary": {n: describe(r, "plume_ssim", pairs=trained) for n, r in b.items() if r},
           "setting_mean_B": float(sm[sm["pair"].isin(trained)]["plume_ssim"].mean()),
           "n_cases_per_seed": int(len(next(iter(b["v0_raw_channels"].values()))))}
    for key, name in (("O6a_Bv0_minus_B_setting_mean", "v0_raw_channels"), ("O6b_Bkonly_minus_B_setting_mean", "k_only")):
        r = contrast(b[name], {s: sm for s in b[name]}, "plume_ssim", pairs=trained)
        r["decision"] = ("SURROGATE BETTER" if r["ci_low"] > 0 and r["p_signflip"] < 0.05 else
                         "BASELINE BETTER" if r["ci_high"] < 0 and r["p_signflip"] < 0.05 else "INCONCLUSIVE")
        out[key] = r
    out["O4b_Bv1_minus_Bv0_withheld"] = contrast(b["v1_factorized_levels"], b["v0_raw_channels"], "plume_ssim",
                                                 rule="superiority_strict_negative", pairs=[PRIMARY_PAIR])
    return out


def _e2_runs(D, root: str, name: str, ckpt: str) -> dict:
    saved = D.pool_root
    D.pool_root = Path(root)
    try:
        return D.runs(f"compositional/{name}", "poolT", ckpt)
    finally:
        D.pool_root = saved


def engineering_summary(D, ckpt: str) -> dict:
    """Descriptive O5b summary: engineering metrics of matched / K-only checkpoints and of the setting mean."""
    def load(rel):
        frames = []
        for sd in sorted((D.pool_root / rel).glob("seed*")):
            f = sd / "e1_fresh_seed_pool_T" / ckpt / "engineering_metrics.csv"
            if f.is_file():
                frames.append(pd.read_csv(f).assign(seed=int(sd.name[4:])))
        return pd.concat(frames) if frames else None
    def summ(df):
        if df is None:
            return None
        res = {k: float(df[k].mean()) for k in ("iou_0.001", "iou_0.0001", "area_rel_err_1e-3_d365", "centroid_err_m_d365")}
        wells = sorted({c[len("exceed_"):-len("_true")] for c in df.columns if c.endswith("_true")})
        t = np.concatenate([df[f"exceed_{w}_true"].to_numpy() for w in wells])
        p = np.concatenate([df[f"exceed_{w}_pred"].to_numpy() for w in wells])
        tp, tn = int(((t == 1) & (p == 1)).sum()), int(((t == 0) & (p == 0)).sum())
        fp, fn = int(((t == 0) & (p == 1)).sum()), int(((t == 1) & (p == 0)).sum())
        res.update({"well_decisions": len(t), "accuracy": (tp + tn) / len(t),
                    "balanced_accuracy": 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1)),
                    "false_negative_rate": fn / max(tp + fn, 1), "false_positive_rate": fp / max(tn + fp, 1),
                    "true_exceedance_rate": float(t.mean())})
        return res
    out = {"matched": summ(load(f"{CONF}/cta_ffl_matched")), "k_only": summ(load(f"{CONF}/cta_ffl_k_only"))}
    if D.setting_mean_csv and "iou_0.001" in pd.read_csv(D.setting_mean_csv, nrows=1).columns:
        out["setting_mean"] = summ(pd.read_csv(D.setting_mean_csv))
    return out


def _unit_means(runs: dict, pairs) -> list[tuple]:
    """(cell, realization, value) with seeds averaged per case first, then transport settings within unit."""
    frames = []
    for s, df in runs.items():
        m = df[df["pair"].isin(pairs)][["param_folder", "realization", "cell", "plume_ssim"]].copy()
        m["seed"] = s
        frames.append(m)
    a = pd.concat(frames)
    case = a.groupby(["param_folder", "realization", "cell"])["plume_ssim"].mean().reset_index()
    unit = case.groupby(["cell", "realization"])["plume_ssim"].mean().reset_index()
    return [(r.cell, r.realization, r.plume_ssim) for r in unit.itertuples()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--master_root", required=True)
    ap.add_argument("--poolT_root", default="")
    ap.add_argument("--registry_v7", required=True)
    ap.add_argument("--fresh_registry", default="")
    ap.add_argument("--seed_classes", default="", help="randf_seed_classes.json (deviation D6)")
    ap.add_argument("--setting_mean_csv", default="", help="per-case metrics of the setting-mean predictor on Pool T")
    ap.add_argument("--split", required=True)
    ap.add_argument("--solver_qa", required=True)
    ap.add_argument("--sections", default="grouped")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    D = Data(args)
    result = {}
    if "grouped" in args.sections:
        result["grouped"] = grouped_section(D)
    if "pool" in args.sections:
        result["poolT_best"] = pool_section(D, "best")
        result["poolT_last"] = pool_section(D, "last")
    Path(args.out).write_text(json.dumps(result, indent=1, default=str))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
