#!/usr/bin/env python3
"""MS-TMO cross-architecture compositional contrast: factorized (V1) vs raw channels (V0).

Supplies the control the v5 manuscript declared missing ("the MS-TMO compositional arm has no
matched raw-channel LOCO control"). Converts that arm from an absolute zero-shot replication into
a genuine same-architecture compositional contrast on the withheld pair.

Deliberately imports `read_variant` and `bootstrap` from compositional_bootstrap.py rather than
reimplementing them, so this contrast uses byte-identical inference machinery to the CTA-UNet+FFL
result: hierarchical resampling of training seeds and geological-factor-by-realization K blocks,
10,000 replicates, percentile 95% CI on direction-adjusted paired differences, p floored at
1/replicates.

Both arms MUST share split and SSIM convention. Pairing across protocols is exactly the defect
corrected on 2026-08-10 (see OBJ1_MSTMO_DELTA_DEFECT_FIX_2026-08-10.md); this script therefore
verifies provenance from each run's config.json before computing anything.

Usage:
    python mstmo_compositional_bootstrap.py <factorized_root> <raw_root> <registry_csv> \
        [--replicates N] [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from compositional_bootstrap import HELDOUT, bootstrap, load_registry, read_variant

EXPECTED = {
    "factorized": "ms_tmo_regime_conditioned",
    "raw": "ms_tmo_transport_conditioned",
}


def verify_provenance(root: Path, arm: str) -> dict:
    """Refuse to compare arms that differ in anything other than the conditioning variant."""
    seen = []
    for seed_dir in sorted(root.glob("seed*")):
        cfg_path = seed_dir / "config.json"
        if not cfg_path.is_file():
            raise SystemExit(f"STOP: missing config.json in {seed_dir}")
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if cfg.get("model_family") != EXPECTED[arm]:
            raise SystemExit(
                f"STOP: {seed_dir} model_family={cfg.get('model_family')!r}, expected {EXPECTED[arm]!r}"
            )
        seen.append(
            {
                "seed": int(seed_dir.name.replace("seed", "")),
                "split_json": Path(str(cfg.get("split_json", ""))).name,
                "stats_json": Path(str(cfg.get("stats_json", ""))).name,
                "epochs": cfg.get("epochs"),
                "batch_size": cfg.get("batch_size"),
                "patch_size": cfg.get("patch_size"),
                "lr": cfg.get("lr"),
            }
        )
    if not seen:
        raise SystemExit(f"STOP: no seed directories under {root}")
    shared = {k: seen[0][k] for k in ("split_json", "stats_json", "epochs", "batch_size", "patch_size", "lr")}
    for entry in seen[1:]:
        for key, value in shared.items():
            if entry[key] != value:
                raise SystemExit(f"STOP: {arm} arm inconsistent on {key}: {value!r} vs {entry[key]!r}")
    return {"arm": arm, "model_family": EXPECTED[arm], "n_seeds": len(seen), **shared}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("factorized_root", help=".../runs_mstmo_crossarch_full_20260807/compositional")
    ap.add_argument("raw_root", help=".../runs_mstmo_raw_loco_full_20260810/compositional")
    ap.add_argument("registry")
    ap.add_argument("--replicates", type=int, default=10000)
    ap.add_argument("--rng-seed", type=int, default=20260810)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    fac_root, raw_root = Path(args.factorized_root), Path(args.raw_root)
    prov_fac = verify_provenance(fac_root, "factorized")
    prov_raw = verify_provenance(raw_root, "raw")

    for key in ("split_json", "stats_json", "epochs", "batch_size", "patch_size", "lr"):
        if prov_fac[key] != prov_raw[key]:
            raise SystemExit(
                f"STOP: arms differ on {key}: factorized={prov_fac[key]!r} raw={prov_raw[key]!r}. "
                "A paired claim requires both arms to share split and protocol."
            )

    combo_of, block_of = load_registry(Path(args.registry))
    target = HELDOUT["compositional"]
    fac = read_variant(fac_root, combo_of, target)
    raw = read_variant(raw_root, combo_of, target)
    if not fac or not raw:
        raise SystemExit("STOP: one arm produced no withheld-pair cases")

    res = bootstrap(fac, raw, block_of, args.replicates, args.rng_seed)
    if res is None:
        raise SystemExit("STOP: no overlapping (seed, param_folder, realization) keys")

    payload = {
        "contrast": "ms_tmo v1_factorized_levels minus v0_raw_channels",
        "split": "compositional",
        "withheld_pair": {"alpha_L": target[0], "alpha_T_ratio": target[1]},
        "replicates": args.replicates,
        "rng_seed": args.rng_seed,
        "p_value_floor": 1.0 / args.replicates,
        "provenance": {"factorized": prov_fac, "raw": prov_raw},
        "results": res,
    }
    text = json.dumps(payload, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)

    for metric, r in res.items():
        if r.get("status") == "incomplete_pairing":
            print(f"WARNING: {metric} has incomplete pairing", file=sys.stderr)


if __name__ == "__main__":
    main()
