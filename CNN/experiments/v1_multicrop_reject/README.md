# V1 Multicrop + Reject

This experiment improves real-world robustness by adding multi-crop inference and
stricter reject/escalate logic. The goal is to reduce wrong-class errors and
only accept confident predictions.

## What V1 Adds (Compared to Baseline)
1) **Multi-crop inference**  
   If the first prediction is not confident, V1 re-checks the image with extra
   crops (zoom center + full resize). It then combines results.

2) **Reject / Escalate logic**  
   If confidence is low, margin is small, or label is already uncertain, V1
   escalates and optionally returns `other_uncertain`.

3) **Stricter rules for risky classes**  
   Example: plastic/glass can require higher confidence.

4) **Selective accuracy**  
   We measure accuracy on *non-rejected* predictions to hit ≥ 95%.

## Important: Training is Unchanged
V1 is **inference-only**. Training stays in:
`CNN/classification_6cats_new/`

Train command:
```
python CNN/classification_6cats_new/train.py --config CNN/classification_6cats_new/configs/train.yaml
```

After training, make sure these files exist (or update paths.yaml):
- `CNN/models/tier1.onnx`
- `CNN/models/label_map.json`
- `CNN/models/infer_config.json`

## Step-by-step (Beginner Friendly)

### Step 0: Check configs
Open:
- `configs/paths.yaml` → paths for model + data + outputs
- `configs/infer_v1.yaml` → thresholds and multicrop settings

### Step 1: (Optional) Train a new model
If you already have a working ONNX model, you can skip this.
```
python CNN/classification_6cats_new/train.py --config CNN/classification_6cats_new/configs/train.yaml
python CNN/classification_6cats_new/export_onnx.py
```

### Step 2: Run evaluation on your real images (G3_SGData)
This compares baseline vs V1 and suggests better thresholds.
```
python CNN/experiments/v1_multicrop_reject/scripts/run_eval_custom_v1.py
```

Outputs:
- `CNN/experiments/v1_multicrop_reject/outputs/eval_custom_v1.json`
- `CNN/experiments/v1_multicrop_reject/outputs/eval_custom_v1.csv`

### Step 3: Update thresholds (if needed)
Open `configs/infer_v1.yaml` and adjust:
- `thresholds.conf`
- `thresholds.margin`
- `thresholds.strict_per_class` (for plastic/glass)
- `multicrop.*` options

### Step 4: Run server (if needed)
```
python CNN/experiments/v1_multicrop_reject/scripts/run_serve_v1.py
```

## Entropy (Yes, it is supported)
Entropy is optional and **disabled by default**.  
To enable:
1) In `configs/infer_v1.yaml`, set:
```
thresholds:
  entropy: 1.5
```
2) Larger entropy means more uncertainty → more rejections.

## Config Quick Reference
- `configs/infer_v1.yaml`
  - `thresholds.conf`: base confidence threshold
  - `thresholds.margin`: top1 - top2 margin
  - `thresholds.entropy`: optional uncertainty threshold
  - `thresholds.strict_per_class`: stricter limits for specific labels
  - `multicrop.enabled`: true/false
  - `multicrop.extra_crops`: [zoom_center, full_resize]
  - `multicrop.combine`: `avg` or `best_score`

- `configs/paths.yaml`
  - `models.onnx`: path to ONNX model
  - `models.label_map`: label list
  - `models.infer_config`: normalization + img_size
  - `data.g3_sgdata`: your real image folder
  - `outputs.base_dir`: where eval results go
