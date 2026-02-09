# V3 Generalization Upgrade (Tier-1)

This v3 package is a low-risk upgrade built on top of your current `v2` trainer.

It does **not** replace v2 code. It adds:
- a stronger v3 training config (`img_size=256`, domain-focused augmentation)
- a train orchestrator (`train_v3.py`) that also runs post-train calibration
- a calibration + threshold evaluation script (`eval_v3_calibrate.py`)
- a baseline-vs-v3 comparison wrapper (`eval_compare_v3.py`)

## Why v3

Your current issue is mostly: **wrong predictions with high confidence** in real photos.

This v3 flow targets that directly:
1. Train with a stronger domain-robust recipe.
2. Add focal loss and Phase A domain-mix to reduce hardset gap.
3. Automatically run temperature scaling + reject-threshold sweep after training.
4. Write recommended thresholds back to `CNN/models/infer_config.json` for deployment.

## Files

- `configs/train_v3.yaml`
- `configs/eval_v3.yaml`
- `scripts/train_v3.py`
- `scripts/eval_compare_v3.py`
- `scripts/eval_v3_calibrate.py`

## 1) Train v3

```powershell
python CNN/experiments/v3_generalization_upgrade/scripts/train_v3.py
```

Optional:

```powershell
python CNN/experiments/v3_generalization_upgrade/scripts/train_v3.py --config CNN/experiments/v3_generalization_upgrade/configs/train_v3.yaml
```

What happens:
- calls `v2` training engine using v3 config
- exports `best.pt` + `model.onnx`
- updates `CNN/models/tier1_best.pt` and `CNN/models/tier1.onnx`
- runs calibration script automatically
- writes calibrated thresholds and temperature into both artifact + `CNN/models/infer_config.json`
- writes run artifacts under `CNN/experiments/v3_generalization_upgrade/artifacts/`

## 2) Compare baseline vs v3

```powershell
python CNN/experiments/v3_generalization_upgrade/scripts/eval_compare_v3.py
```

Outputs go to:
- `CNN/experiments/v3_generalization_upgrade/outputs/eval_compare_v2_<timestamp>/`

## 3) Run calibration + threshold recommendation

```powershell
python CNN/experiments/v3_generalization_upgrade/scripts/eval_v3_calibrate.py
```

Outputs go to:
- `CNN/experiments/v3_generalization_upgrade/outputs/v3_eval_<timestamp>/eval_v3_calibration.json`
- `CNN/experiments/v3_generalization_upgrade/outputs/v3_eval_<timestamp>/threshold_sweep.csv`
- confusion and reliability plots in same folder

Key outputs to watch:
- `metrics_raw` vs `metrics_calibrated`
- `temperature_scaling.temperature`
- `recommended_reject_thresholds`
- `coverage` and `selective_acc`

## Suggested 1-3 day experiment order

1. Train v3 once with default config.
2. Run `eval_compare_v3.py` and keep the best run.
3. Run `eval_v3_calibrate.py` on that model.
4. Apply recommended reject thresholds to your v1/v2 reject config.
5. Re-test Android real scans with 20-50 hard photos.

## Notes

- v3 config uses `onnx.opset: 11` for better compatibility on older Android devices.
- If export fails on your environment, set opset to `13` in `train_v3.yaml` and rerun.
