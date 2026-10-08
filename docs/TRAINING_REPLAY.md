# Historical and portable validation crops

No training replay is performed by CPU CI. Full replay needs authorized data and a reviewed GPU environment.
The injection well is at row/column `(200,199)`. The validation crop anchor is `(399,199)`, 199 rows downstream.
Legacy `source_row/source_col` configuration fields denote the anchor, not the injection location.

The backward-compatible trainer accepts `--validation_anchor_row 399 --validation_anchor_col 199` and the
older aliases. Future crop manifests label these as validation anchors. Coordinates and crop algorithms are
unchanged; historical manifests/configs are preserved without relabeling. Each of the 60 shipped historical
manifests has 84 crops: all contain the anchor, but only 68 contain the injection well.

For authorized replay only, the command forms are:

```bash
# Historical mode: identical absolute file names are needed to reproduce hash-derived validation crops.
python -m src.transport_surrogates.confirmatory_v5.train --data_root "$AUTHORIZED_DATA_ROOT" --val_crop_key absolute_legacy --validation_anchor_row 399 --validation_anchor_col 199 --out_root "$NEW_OUTPUT_ROOT"
# Portable mode: relative sample identifiers are hashed, so historical crop origins may differ.
python -m src.transport_surrogates.confirmatory_v5.train --data_root "$AUTHORIZED_DATA_ROOT" --val_crop_key relative --validation_anchor_row 399 --validation_anchor_col 199 --out_root "$NEW_OUTPUT_ROOT"
```

These are mode examples, not complete scientific training recipes or launch commands. Supply the recorded
split, normalization, exclusions, conditioning, model and optimizer settings for the chosen run. Do not
overwrite an original output. Exact GPU trajectories, alternative source-containing checkpoint selection and
data-dependent inference were not verified from this release.
