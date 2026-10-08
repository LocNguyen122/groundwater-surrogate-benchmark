"""Reproduce summary-derived macros and contrast tables in a new output directory."""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from verify_snapshot import verify, lf_text_sha256

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    verify()
    if a.output.exists():
        raise FileExistsError("Reproduction output must be new")
    a.output.mkdir(parents=True)
    out = a.output / "numbers_v7.tex"
    subprocess.run([sys.executable, str(ROOT / "scripts/v7_analysis/make_numbers_v7.py"),
                    str(ROOT / "scripts/v7_analysis/inputs_v7.json"), str(out.resolve())], check=True)
    record = json.loads((ROOT / "provenance/PACKAGING_RECORD.json").read_text())
    if lf_text_sha256(out.read_bytes()) != record["numeric_macro_lf_sha256"]:
        raise ValueError("Numeric macro reproduction differs from the verified source")
    rows = []
    for ck in ("best", "last"):
        j = json.loads((ROOT / f"results/v7/analysis/v7_poolT_{ck}_terminal_2026-10-08.json").read_text())
        contrasts = {"baseline": j["O5a_matched_minus_setting_mean"],
                     "primary": j["O3_compositional"]["primary_v1_minus_v0"],
                     **j["O3_compositional"]["loco_secondary"]}
        for label, r in contrasts.items():
            rows.append({"checkpoint": ck, "contrast": label, **{k: r.get(k, "") for k in
                         ("estimate", "ci_low", "ci_high", "p_signflip", "p_holm", "n_seeds", "n_cells")}})
    with (a.output / "checkpoint_contrasts.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("PASS: identical source-derived numeric macros and", len(rows), "contrast rows")
