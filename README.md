# Groundwater surrogate benchmark and evaluation audit

Private research-code snapshot for reviewing deterministic groundwater plume surrogates and their evaluation
on independently seeded geology. This is **not a final paper release or an open-source release**. The associated
manuscript is an unpublished review draft; no article DOI or acceptance is claimed. See [release status](RELEASE_STATUS.json).

## Research question and methods

Can high accuracy on grouped simulation data establish prediction for new conductivity fields? This study audits
seed reuse and rescaled conductivity twins, compares a conductivity-free setting-mean baseline, and examines
transport-code conditioning under withheld combinations. Implementations include CTA-UNet, the related MS-TMO
variant, CNN/Pix2Pix and operator reference models. Current statistical analysis uses paired random-stream clusters,
hierarchical bootstrap intervals, sign-flip tests and multiplicity correction.

Scientific interpretation matters: original grouped fields have training twins. On independently seeded geology,
matched-code plume SSIM is approximately 0.512, below the setting mean of 0.822. The primary conditioning contrast
remains positive, but the five-seed secondary `(62, 1)` contrast is inconclusive at both selected and final
checkpoints. The related MS-TMO arm is not architecture-independent replication. Older results are preserved as
historical measurements, not relabeled as evidence of independent-geology generalization.

## CPU-only quick start

Use Python 3.11 in an isolated environment. CI installs CPU-only PyTorch and does not run training, dataset
generation, GPU evaluation or simulator commands.

```bash
python -m venv .venv
# Activate .venv using the command for your operating system.
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-cpu.txt
python scripts/verify_snapshot.py
python scripts/run_release_tests.py
python scripts/validate_release_artifacts.py
python scripts/reproduce_summaries.py --output reproduced
```

`reproduced/` must be new. The final command reproduces the numeric macro bundle and exports checkpoint contrasts
from the shipped statistical summaries. It verifies hashes before running and does not overwrite numerical sources.
For full 10,000-replicate recomputation from the permitted per-case CSVs:

```bash
python scripts/v7_analysis/regenerate_checkpoint.py --checkpoint best --output reproduced/best.json
python scripts/v7_analysis/regenerate_checkpoint.py --checkpoint last --output reproduced/last.json
```

These longer CPU calculations require neither simulation fields nor checkpoints. Optional engineering output can
differ from a scoring-only summary when the original engineering CSVs are available; the SSIM summaries and
inferential contrasts are unchanged. Absolute means and equal-cluster-weighted contrasts use different weighting.

## Layout

- `src/`: paper-specific models, trainers and necessary shared transforms/metrics/evaluation utilities.
- `scripts/`: metric audits, statistics, table/figure utilities and snapshot checks.
- `tests/`: synthetic CPU-safe regression tests; no restricted assets required.
- `configs/`, `splits/`, `metadata/`: training statistics, fixed splits, control mappings and permitted scientific metadata.
- `results/`: lightweight per-case and aggregate measurements, including historical results and current certifications.
- `figures/v7/`: numeric figure provenance only, not simulation arrays or manuscript figures.
- `provenance/`: source file checksums and a record of snapshot-only edits.

Historical run identifiers inside result paths are preserved to keep scientific traceability. No development
checkout, compute-cluster workspace or inherited Git history is included.

## Availability, integrity and limitations

Raw fields, simulation setups/commands/executables, checkpoints and prediction caches are excluded. Full training,
field-based evaluation and inference-dependent figures require separately authorized assets and have **not** been
tested from this repository. Only trusted checkpoint files should ever be loaded. Restricted data access is not
granted by possession of this code. See [scientific protocol and disclosures](docs/SCIENTIFIC_PROTOCOL.md).

The selected- and final-checkpoint scoring campaigns each completed 250 tasks successfully. This closes a scoring
gate, not manuscript approval. Original grouped training suppressed 224 AMP updates; independent retraining
suppressed 70. The older trainer sanitized nonfinite raw outputs, so finite losses do not prove finite raw outputs.
Recovered-artifact and failure disclosures remain in their original scientific summaries.

MIT is the preferred candidate for original code, but ownership and license authorization remain unconfirmed.
No open-source license is applied. Dependencies keep their own terms; see [license status](LICENSE_STATUS.md) and
[third-party notices](THIRD_PARTY_NOTICES.md). Material AI assistance in code refactoring, analysis tooling and
language editing is disclosed in the scientific protocol. It does not replace human responsibility for claims.

The GitHub workflow uses read-only permissions and immutable action revisions, following
[GitHub's secure-use guidance](https://docs.github.com/en/actions/reference/security/secure-use). CPU installation
commands follow the [official PyTorch version instructions](https://pytorch.org/get-started/previous-versions/).
Historical training used its recorded environment; CI is a separate compatibility check, not a training rerun.

## Citation and review

`CITATION.cff` identifies this software snapshot without inventing an article DOI or publication. The associated
manuscript is titled *Hidden Conductivity Twins Inflate the Accuracy of Groundwater Plume Surrogates: An Audit
on Independently Seeded Geology*. Manuscript approval, code licensing, confirmed data-access terms and venue
choice remain pending. Private repository access must be granted separately before others can review it.
