# V1 Robust Fine-tune

This experiment adds a **two-phase fine-tuning** strategy + stronger
augmentations to improve real-world robustness (small objects + cluttered
backgrounds). It is designed to be safe and fast to iterate.

## What It Does
1) **Phase A (general)**: trains on mixed dataset splits (tier1_splits)
2) **Phase B (domain)**: fine-tunes on G3_SGData with lower LR
3) **Robust augmentations**:
   - ScaleDownPad (zoom-out + pad)
   - Strong RandomResizedCrop (low scale)
   - MixUp + CutMix
   - ColorJitter, RandomAffine/Perspective, RandomErasing
4) **Exports** a new ONNX model and runs a quick ORT sanity check
5) **Evaluation** compares baseline vs robust model on G3_SGData

## Backup Copy
Original training + config snapshots live in:
`backup_clf_7cats_tier1/`

## 1) Train Robust V1
```
python CNN/experiments/v1_robust_finetune/scripts/train_v1.py \
  --config CNN/experiments/v1_robust_finetune/configs/train_v1.yaml
```

Outputs:
- `CNN/experiments/v1_robust_finetune/outputs/<run_name>/`
- `CNN/experiments/v1_robust_finetune/artifacts/<run_name>/best.pt`
- `CNN/experiments/v1_robust_finetune/artifacts/<run_name>/infer_config.json`
- `CNN/experiments/v1_robust_finetune/artifacts/<run_name>/label_map.json`
- `CNN/experiments/v1_robust_finetune/artifacts/<run_name>/run_summary.json`

## 2) Export to ONNX
```
python CNN/experiments/v1_robust_finetune/scripts/export_v1.py \
  --config CNN/experiments/v1_robust_finetune/configs/train_v1.yaml
```

This will export to:
- `CNN/experiments/v1_robust_finetune/artifacts/<run_name>/model.onnx`

## 3) Compare Eval (Baseline vs Robust)
```
python CNN/experiments/v1_robust_finetune/scripts/eval_compare_v1.py
```

Outputs:
- `CNN/experiments/v1_robust_finetune/outputs/eval_compare_v1_<timestamp>/eval_compare_v1.json`
- `CNN/experiments/v1_robust_finetune/outputs/eval_compare_v1_<timestamp>/eval_compare_v1.csv`
- Confusion matrices for baseline/robust per group

## Config Notes
- `configs/train_v1.yaml` controls training + augmentations.
- `phase_a/phase_b` control epochs and LRs.
- `augment.scale_down_pad` simulates small objects in clutter.
- `mixup_cutmix` enables MixUp/CutMix (configurable).
- Domain data is read from `paths.g3_data_dir`.

## Tips
- If robust model doesn¡¯t improve, check:
  - ScaleDownPad enabled
  - MixUp/CutMix enabled
  - Phase B LR lower than Phase A
  - Domain split ratio reasonable (default 0.2)
