# Bin-It-Right ML (CNN)

This folder contains active CNN model code used by the product pipeline.

## Active folders
- `clf_7cats_tier1/`: current main training/inference pipeline
- `experiments/`: active experiments (v1 reject/multicrop, v3 upgrade)
- `shared/`: reusable preprocessing/decision/onnx helpers
- `services/`: scan API contract + service-side logic
- `models/`: promoted runtime artifacts
- `docs/`: technical integration docs

## Archived code
Older baseline files are moved to:
- `legacy/cnn/legacy_root/`

Archived experiments and datasets are also under `legacy/cnn/`.

## Note
`v1_parity_self_test` stays in active experiments because it is still useful for Android/Python preprocessing parity checks.
