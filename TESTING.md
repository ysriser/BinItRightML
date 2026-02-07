# Testing Guide (ML)

## Unit tests we keep as critical
- `test_main.py`: API `/forecast` contract (success + missing data).
- `CNN/services/tests/test_scan_service_v0_1_contract.py`: `/api/v1/scan` v0.1 contract and Tier-2 fallback behavior.
- `CNN/services/tests/test_scan_service_v0_1_thresholds.py`: Tier-2 trigger thresholds (including strict class thresholds).
- `CNN/shared/tests/test_decision.py` and `CNN/shared/tests/test_multicrop.py`: shared inference logic.
- `CNN/clf_7cats_tier1/tests/test_manifest_utils.py` and `CNN/clf_7cats_tier1/tests/test_serve_api.py`: data/serve critical paths.

## Local command
```bash
python -m pytest -q test_main.py CNN/shared/tests CNN/clf_7cats_tier1/tests CNN/services/tests
```

## CI evidence
- Workflow: `.github/workflows/pr_validation.yml`
- Artifacts:
  - `python-unit-reports`
  - `python-lint-reports`
  - `python-security-testing-report`
