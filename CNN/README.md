# Bin-It-Right ML

This repo focuses on **image classification**. We keep datasets and experiments
separated to avoid config drift.

## Common folders
- `CNN/data/` (datasets, not committed)
- `CNN/outputs/` (training outputs)
- `CNN/models/` (exported models)
- `CNN/experiments/` (versioned experiments)
- `CNN/shared/` (reusable inference utilities)
- `CNN/legacy_root/` (older root-level files, kept for reference)

## Key pipelines
- Training: `CNN/clf_7cats_tier1/`
- V1 inference logic: `CNN/experiments/v1_multicrop_reject/`
- Parity self-test: `CNN/experiments/v1_parity_self_test/`
- Robust fine-tune: `CNN/experiments/v1_robust_finetune/`

## Parity self-test (quick check)
```
python CNN/experiments/v1_parity_self_test/parity_cli.py --mode golden --images CNN/experiments/v1_parity_self_test/samples
```
