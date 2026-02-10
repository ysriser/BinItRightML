# Forecast Module

This folder contains the forecasting API and load-test entrypoints.

## Contents
- `main.py`: FastAPI app for `/forecast`
- `locustfile.py`: Locust load test user profile
- `data/`: Forecast model assets and source data
- `tests/`: Forecast-focused unit tests

## Compatibility
Root-level `main.py`, `locustfile.py`, and `test_main.py` are kept as thin wrappers so existing CI and deployment commands continue to work.
