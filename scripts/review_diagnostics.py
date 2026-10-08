"""Post-review CPU diagnostics from permitted CSVs; never runs models or simulators."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAPS = 10000


def contains(y: int, x: int, row: int, col: int, patch: int = 320) -> bool:
    return y <= row < y + patch and x <= col < x + patch


def crop_audit(paths: list[Path]) -> list[dict]:
    out = []
    for path in paths:
        rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
        if not rows:
            raise ValueError("Empty validation manifest")
        source = sum(contains(int(r['crop_y0']), int(r['crop_x0']), 200, 199) for r in rows)
        anchor = sum(contains(int(r['crop_y0']), int(r['crop_x0']), 399, 199) for r in rows)
        out.append({'manifest_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'rows': len(rows),
                    'injection_well_included': source, 'downstream_anchor_included': anchor,
                    'injection_well_fraction': source / len(rows)})
    return out


def mapping():
    registry = pd.read_csv(ROOT / 'metadata/fresh_seed/param_registry_fresh_seed.csv')
    registry = registry[registry.pool == 'T']
    classes = json.loads((ROOT / 'metadata/fresh_seed/randf_seed_classes.json').read_text())['class_of_seed']
    return {f'param_{int(r.param_id)}': (f'class_{classes[str(int(r.seed))]}', int(r.cell_id))
            for r in registry.itertuples()}


def add_group(frame: pd.DataFrame, groups: dict) -> pd.DataFrame:
    if frame[['param_folder', 'realization']].duplicated().any():
        raise ValueError('Duplicate case keys')
    frame = frame.copy()
    frame['cluster'] = frame.param_folder.map(lambda p: groups[p][0])
    frame['geology'] = frame.param_folder.map(lambda p: groups[p][1])
    return frame


def rates(counts: np.ndarray) -> dict:
    tp, fp, tn, fn = np.moveaxis(counts, -1, 0)
    div = lambda a, b: np.divide(a, b, out=np.full_like(a, np.nan, dtype=float), where=b > 0)
    sensitivity, specificity = div(tp, tp + fn), div(tn, tn + fp)
    return {'sensitivity': sensitivity, 'specificity': specificity,
            'false_negative_rate': 1 - sensitivity, 'false_positive_rate': 1 - specificity,
            'balanced_accuracy': (sensitivity + specificity) / 2}


def well_summary(frames: list[pd.DataFrame], groups: dict, n_boot: int = BOOTSTRAPS) -> dict:
    fs = [add_group(f, groups) for f in frames]
    columns = sorted(c for c in fs[0] if c.startswith('exceed_') and c.endswith('_true'))
    wells = [c[len('exceed_'):-len('_true')] for c in columns]
    clusters = sorted(fs[0].cluster.unique())
    count = np.zeros((len(fs), len(clusters), len(wells), 4), dtype=float)
    for s, frame in enumerate(fs):
        if len(frame) != 794 or set(frame.cluster) != set(clusters):
            raise ValueError('Incomplete engineering case/cluster coverage')
        for g, cluster in enumerate(clusters):
            block = frame[frame.cluster == cluster]
            for w, tag in enumerate(wells):
                a, b = block[f'exceed_{tag}_true'].to_numpy(), block[f'exceed_{tag}_pred'].to_numpy()
                if not np.isin(a, [0, 1]).all() or not np.isin(b, [0, 1]).all():
                    raise ValueError('Nonbinary well decisions')
                count[s, g, w] = [np.sum((a == 1) & (b == 1)), np.sum((a == 0) & (b == 1)),
                                  np.sum((a == 0) & (b == 0)), np.sum((a == 1) & (b == 0))]
    rng = np.random.default_rng(20261009)
    boot = np.empty((n_boot, len(wells), 4))
    for i in range(n_boot):
        sd, gd = rng.integers(len(fs), size=len(fs)), rng.integers(len(clusters), size=len(clusters))
        # One cluster draw keeps all seven wells together, preserving their dependence.
        boot[i] = count[np.ix_(sd, gd)].sum(axis=(0, 1))
    def record(point, draws):
        return {k: {'estimate': float(v), 'ci95': np.nanquantile(rates(draws)[k], [.025, .975]).tolist()}
                for k, v in rates(point).items()}
    pooled = count.sum(axis=(0, 1))
    return {'n_seeds': len(fs), 'n_clusters': len(clusters), 'n_cases_per_seed': 794,
            'n_wells': len(wells), 'bootstrap_replicates': n_boot,
            'interpretation': 'Exploratory case-weighted rates; resample random-stream blocks and training seeds; not a threshold sensitivity or calibration proof',
            'pooled': record(pooled.sum(axis=0), boot.sum(axis=1)),
            'per_well': {tag: {'confusion_pooled_seed_repeats': pooled[w].astype(int).tolist(),
                              **record(pooled[w], boot[:, w])} for w, tag in enumerate(wells)}}


def weighting(frames, baseline, groups, n_boot=BOOTSTRAPS):
    diffs = []
    for seed, f in enumerate(frames):
        m = f.merge(baseline[['param_folder', 'realization', 'plume_ssim']],
                    on=['param_folder', 'realization'], validate='one_to_one', suffixes=('_model', '_mean'))
        if len(m) != 794:
            raise ValueError('Incomplete paired baseline coverage')
        m = add_group(m, groups)
        m['difference'] = m.plume_ssim_model - m.plume_ssim_mean
        m['seed'] = seed
        diffs.append(m)
    frame = pd.concat(diffs, ignore_index=True)
    if not np.isfinite(frame.difference).all():
        raise ValueError('Nonfinite paired difference')
    cells = frame.groupby(['cluster', 'geology']).difference.mean().reset_index()
    clusters = sorted(cells.cluster.unique())
    cluster_mean = cells.groupby('cluster').difference.mean()
    sums = frame.groupby('cluster').difference.agg(['sum', 'count']).reindex(clusters).to_numpy()
    cell_sums = cells.groupby('cluster').difference.agg(['sum', 'count']).reindex(clusters).to_numpy()
    rng = np.random.default_rng(20261009)
    draws = rng.integers(len(clusters), size=(n_boot, len(clusters)))
    case_boot = sums[draws, 0].sum(1) / sums[draws, 1].sum(1)
    cell_boot = cell_sums[draws, 0].sum(1) / cell_sums[draws, 1].sum(1)
    return {'case_weighted': float(frame.difference.mean()), 'equal_geological_cell': float(cells.difference.mean()),
            'equal_stream_descriptive': float(cluster_mean.mean()),
            'case_weighted_cluster_bootstrap_ci95': np.quantile(case_boot, [.025, .975]).tolist(),
            'cell_weighted_cluster_bootstrap_ci95': np.quantile(cell_boot, [.025, .975]).tolist(),
            'n_clusters': len(clusters), 'n_geological_cells': len(cells), 'n_boot': n_boot,
            'interpretation': 'Post-review descriptive sensitivity; cluster-block bootstrap conditional on seed-averaged cases. Does not replace the original three-level primary contrast or confer independent status on 27 cells.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--validation-root', type=Path, default=ROOT / 'results/v8/validation')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output; preserve previous measurements')
    groups = mapping()
    baseline = pd.read_csv(ROOT / 'results/v7/per_case/setting_mean/setting_mean_A_poolT_per_case.csv')
    frames = {name: [pd.read_csv(p) for p in sorted((ROOT / f'results/v8/engineering/{name}').glob('seed*.csv'))]
              for name in ('matched', 'k_only')}
    if any(len(f) != 10 for f in frames.values()):
        raise ValueError('Need ten complete engineering seed CSVs per model')
    result = {'scope': 'Post-review diagnostics, not pre-registered experiments',
              'source_coordinate_row_col': [200, 199], 'validation_anchor_row_col': [399, 199],
              'wells': {name: well_summary(fs, groups) for name, fs in {**frames, 'log_setting_mean': [baseline]}.items()},
              'baseline_weighting': {name: weighting(fs, baseline, groups) for name, fs in frames.items()},
              'validation_manifests': crop_audit(sorted(args.validation_root.rglob('*.csv'))) if args.validation_root else [],
              'source_hashes': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (ROOT / 'results/v8/engineering').glob('*/*.csv')}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'models': list(result['wells']), 'crop_manifests': len(result['validation_manifests'])}))


if __name__ == '__main__':
    main()
