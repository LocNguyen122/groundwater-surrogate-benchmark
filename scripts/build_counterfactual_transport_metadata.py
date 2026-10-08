"""Build six inference-only fixed-pair metadata files for counterfactual sweeps."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "metadata" / "parameter_registry_162_v5.csv"
OUT_DIR = ROOT / "metadata" / "confirmatory_controls_v5" / "counterfactual_sweep"
ALPHA_L = (0.62, 6.2, 62.0)
RATIOS = (0.1, 1.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    with REGISTRY.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "purpose": "Inference-only fixed coefficient-pair sweep using checkpoint training statistics.",
        "files": [],
    }
    for alpha_l in ALPHA_L:
        for ratio in RATIOS:
            alpha_tag = str(alpha_l).replace(".", "p")
            ratio_tag = str(ratio).replace(".", "p")
            path = OUT_DIR / f"fixed_alphaL_{alpha_tag}_ratio_{ratio_tag}.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("param_folder", "alpha_L", "alpha_T_ratio"))
                writer.writeheader()
                for row in sorted(rows, key=lambda item: item["param_folder"]):
                    writer.writerow(
                        {
                            "param_folder": row["param_folder"],
                            "alpha_L": alpha_l,
                            "alpha_T_ratio": ratio,
                        }
                    )
            manifest["files"].append(
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "alpha_L": alpha_l,
                    "alpha_T_ratio": ratio,
                    "sha256": sha256(path),
                }
            )
    manifest_path = OUT_DIR / "COUNTERFACTUAL_SWEEP_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
