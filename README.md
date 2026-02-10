# Bin-It-Right ML

This repository contains the ML services and model pipelines for BinItRight.

## Repository Layout

- `CNN/`: active image classification pipelines and serving utilities
- `forecast/`: forecasting API and load-testing module
- `legacy/`: archived modules and older experiment code kept for traceability
- `.github/workflows/`: CI validation and security scans

## Active CNN Layout

- `CNN/clf_7cats_tier1/`: core 7-category classification pipeline
- `CNN/services/`: scan service API contracts and tests
- `CNN/shared/`: reusable decision and preprocessing utilities
- `CNN/experiments/v1_multicrop_reject/`: runtime reject + multicrop logic
- `CNN/experiments/v3_generalization_upgrade/`: latest training/eval upgrade path
- `CNN/docs/`: integration and API contract docs

`CNN/experiments/v1_parity_self_test/` is intentionally kept active because it validates ONNX preprocessing parity between Python and Android.

## Forecast Layout

- `forecast/main.py`: FastAPI forecast endpoint module
- `forecast/locustfile.py`: locust performance profile
- `forecast/data/`: forecast source data and model artifact files
- `forecast/tests/`: forecast-focused tests

Compatibility wrappers remain at repo root for CI/deploy stability:

- `main.py`
- `locustfile.py`
- `test_main.py`

## Legacy Layout

- `legacy/cnn/`: archived CNN modules and old experiment branches
- `legacy/cnn/legacy_root/`: previous root-level CNN baseline files
- `legacy/forecast/`: placeholder for archived forecast versions

Legacy source code is preserved. Generated outputs are ignored by `.gitignore`.

## Common Commands

```bash
# Run forecast API locally
uvicorn main:app --reload

# Run unit tests used by CI
pytest test_main.py CNN/shared/tests CNN/clf_7cats_tier1/tests CNN/services/tests

```
