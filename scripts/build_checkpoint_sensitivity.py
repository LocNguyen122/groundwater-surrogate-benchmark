"""Plot checkpoint sensitivity from certified summaries; no prediction or data regeneration."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
import matplotlib.pyplot as plt


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--best", type=Path, required=True)
    p.add_argument("--last", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError("Use a new analysis directory")
    a.out.mkdir(parents=True)
    rows = []
    for checkpoint, path in (("selected", a.best), ("final", a.last)):
        j = json.loads(path.read_text())
        contrasts = {"matched minus setting mean": j["O5a_matched_minus_setting_mean"],
                     "primary V1 minus V0": j["O3_compositional"]["primary_v1_minus_v0"],
                     "secondary (62,1)": j["O3_compositional"]["loco_secondary"]["loco_62p0_1p0"],
                     "MS-TMO V1 minus V0": j["O3_compositional"]["mstmo_v1_minus_v0"]}
        for label, r in contrasts.items():
            rows.append({"checkpoint": checkpoint, "contrast": label,
                         **{k: r.get(k, "") for k in ("estimate", "ci_low", "ci_high", "p_signflip",
                                                       "p_holm", "n_seeds", "n_cells", "n_boot")}})
    with (a.out / "checkpoint_contrasts.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for name, labels in (("baseline_gap", ["matched minus setting mean"]),
                         ("conditioning_contrasts", ["primary V1 minus V0", "secondary (62,1)", "MS-TMO V1 minus V0"])):
        fig, ax = plt.subplots(figsize=(7, 2.6 if len(labels) == 1 else 4), constrained_layout=True)
        for i, label in enumerate(labels):
            for ck, offset, color in (("selected", .12, "#2166ac"), ("final", -.12, "#b35806")):
                r = next(r for r in rows if r["checkpoint"] == ck and r["contrast"] == label)
                ax.errorbar(r["estimate"], i + offset,
                            xerr=[[r["estimate"] - r["ci_low"]], [r["ci_high"] - r["estimate"]]],
                            fmt="o", color=color, capsize=3, label=ck if i == 0 else None)
        ax.set_yticks(range(len(labels)), labels)
        ax.axvline(0, color="gray", linewidth=.8)
        ax.set_xlabel("Cluster-weighted plume SSIM difference (95% bootstrap interval)")
        ax.legend(frameon=False)
        for suffix in ("pdf", "png"):
            fig.savefig(a.out / (name + "." + suffix), dpi=180)
        plt.close(fig)
