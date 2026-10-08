# Manuscript reproduction map

The manuscript is an unpublished v8 review draft. The numeric macro bundle keeps the historical name
`numbers_v7.tex` because its 570 source-derived values have not changed. CPU reproduction is not a full
training, field-audit or simulator replication. No article DOI or approved open-source license is claimed.

Run `python scripts/reproduce_summaries.py --output reproduced` from a fresh checkout. This verifies the
manifest, regenerates all 570 values from `scripts/v7_analysis/inputs_v7.json`, checks the normalized text
checksum, and exports selected/final checkpoint contrast rows. Inputs in the configuration are portable.

| Item | Released numeric inputs | Reproduction / limitation |
|---|---|---|
| Table 1, numerical/data protocol | `metadata/`, `splits/`, protocol documentation | Specification, not simulator execution; model inputs/executables excluded |
| Table 2, grouped benchmark | `results/v7/analysis/v7_grouped_results.json` | Macro command above; original per-case inputs retained |
| Table 3, independent absolute scores | `results/v7/analysis/v7_poolT_best_complete_2026-10-08.json` | Macro command; `build_review_figures.py` generates added mean/SD text |
| Table 4, engineering quantities | `results/v7/analysis/v7_poolT_engineering_O5b.json` | Macro command; engineering CSVs in `results/v8/engineering` support well-rate diagnostics |
| Table 5, conditioning contrasts | grouped JSON and selected/final terminal JSON under `results/v7/analysis` | Macro command; per-case CSVs retained for paired statistics |
| Table 6, grouped withheld-pair screen | `v7_grouped_results.json` and pilot sensitivity JSON | Macro command; exploratory after retained-pilot selection |
| Table 7, independent secondary screen | selected/final terminal JSON | Macro command; `(62,1)` remains inconclusive |
| Table 8, retraining | `v7_e2_poolT_results.json` | Macro command; added sample SD generated from the same summary |
| Table 9, training settings | recorded configs and Appendix A | Documentary specification; exact historical GPU/solver environment is incomplete |
| Figure 1, workflow | original vector layout in `scripts/build_review_figures.py` | `python scripts/build_review_figures.py --output reproduced/figures`; no empirical field pixels |
| Figure 2, counterfactual sweep | `figures/v7/fig_v7_counterfactual.json` pooled matrices and consistency summaries | `build_review_figures.py` redraws the pooled matrix; original per-case sweep CSVs are not included, so seed/cell consistency cannot be independently reaggregated from this snapshot |
| Figure 3, well trade-offs | 20 engineering CSVs, setting-mean per-case CSV, fresh-seed registry | Commands below; conditional exploratory block intervals, not a formal superiority test |

```bash
python scripts/review_diagnostics.py --output reproduced/review_diagnostics.json
python scripts/build_review_figures.py --output reproduced/figures
```

The diagnostic command uses 10,000 random-stream/seed block resamples and keeps the seven wells together.
Per-well confusion counts pool seed repetitions, not independent new simulation observations. Continuous well
concentrations are not released, so this cannot reproduce threshold sensitivity. Equal-case/cell descriptive
weightings do not replace equal-stream primary contrasts or make 27 cells independent.

The previous empirical one-case conductivity/plume illustration is withdrawn from v8: its conductivity panels
used inconsistent color limits. The builder now enforces common limits, but the actual empirical figure was
not regenerated from restricted predictions. It is not evidence in this version. No new model scores were run.

Hashes and per-case metrics enable numerical consistency checks, not independent verification of actual
conductivity twins, generator seed collapse or solver overshoots. That requires rights-holder-authorized fields
and generator provenance. CPU CI does not close that scientific access gate.
