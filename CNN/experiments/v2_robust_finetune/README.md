# V2 Robust Fine-tune

This experiment adds a two-phase fine-tuning strategy plus stronger
augmentations to improve real-world robustness (small objects + cluttered
backgrounds). It is designed to be safe and fast to iterate.

## What It Does
1) Phase A (general): trains on mixed dataset splits (tier1_splits)
2) Phase B (domain): fine-tunes on CNN/data/hardset with lower LR
3) Robust augmentations:
   - ScaleDownPad (zoom-out + pad)
   - Strong RandomResizedCrop (low scale)
   - MixUp + CutMix
   - ColorJitter, RandomAffine/Perspective, RandomErasing
4) Exports a new ONNX model and runs a quick ORT sanity check
5) Evaluation compares baseline vs robust model on hardset

## Backup Copy
Original training + config snapshots live in:
backup_clf_7cats_tier1/

## 1) Train Robust V2
```
python CNN/experiments/v2_robust_finetune/scripts/train_v1.py \
  --config CNN/experiments/v2_robust_finetune/configs/train_v2.yaml
```

Outputs:
- CNN/experiments/v2_robust_finetune/outputs/<run_name>/
- CNN/experiments/v2_robust_finetune/artifacts/<run_name>/best.pt
- CNN/experiments/v2_robust_finetune/artifacts/<run_name>/infer_config.json
- CNN/experiments/v2_robust_finetune/artifacts/<run_name>/label_map.json
- CNN/experiments/v2_robust_finetune/artifacts/<run_name>/run_summary.json

## 2) Export to ONNX
```
python CNN/experiments/v2_robust_finetune/scripts/export_v1.py \
  --config CNN/experiments/v2_robust_finetune/configs/train_v2.yaml
```

This will export to:
- CNN/experiments/v2_robust_finetune/artifacts/<run_name>/model.onnx

## 3) Compare Eval (Baseline vs Robust)
```
python CNN/experiments/v2_robust_finetune/scripts/eval_compare_v1.py
```

Outputs:
- CNN/experiments/v2_robust_finetune/outputs/eval_compare_v1_<timestamp>/eval_compare_v1.json
- CNN/experiments/v2_robust_finetune/outputs/eval_compare_v1_<timestamp>/eval_compare_v1.csv
- Confusion matrices for baseline/robust per group

## Config Notes (Training + Augmentations)

### Loss
- `loss.label_smoothing`
  - What: Softens hard labels (reduces over-confidence).
  - Impact: Helps generalization + fewer “confident wrong” cases (good for reject).
  - Suggested: `0.05` (try `0.07` if still too confident; back to `0.05` if coverage drops too much).

### Augment (Phase A / General)
- `augment.rrc_p`
  - What: Probability of `RandomResizedCrop`.
  - Impact: Robust to framing/partial views; too high can add label noise.
  - Suggested: `1.0` in Phase A is fine; lower in Phase B (you do `0.7`).

- `augment.rrc_scale_min`, `augment.rrc_scale_max`
  - What: Crop area range (as fraction of original image area).
  - Impact: Lower `min` = more aggressive crops (better robustness, more noise).
  - Suggested: `min=0.4–0.6` (your `0.5` is safe). If complex scenes still weak, try `0.35–0.45`.

- `augment.rrc_ratio`
  - What: Crop aspect ratio range.
  - Impact: Robust to portrait/landscape framing.
  - Suggested: `[0.75, 1.33]` (keep).

- `augment.scale_down_pad.enabled`
  - What: Enable “zoom-out then pad” (simulate small objects in frame).
  - Impact: Usually the biggest boost for small-object + clutter cases.
  - Suggested: `true` (keep on).

- `augment.scale_down_pad.prob`
  - What: Probability of ScaleDownPad.
  - Impact: Higher = more small-object training; can reduce “clean” accuracy if too high.
  - Suggested: `0.3` is a safe start. If complex still low, try `0.4–0.6`.

- `augment.scale_down_pad.scale_range`
  - What: Zoom-out scale range. Smaller = harder (object becomes smaller).
  - Impact: Main knob for small-object robustness; too small adds noisy samples.
  - Suggested: `[0.6, 0.9]` is mild. If complex scenes are a priority, try `[0.4, 0.8]` or `[0.45, 0.75]`.

- `augment.scale_down_pad.random_position`
  - What: Place the zoomed object at random positions.
  - Impact: Robust to off-center objects.
  - Suggested: `true` (keep).

- `augment.scale_down_pad.pad_mode`
  - What: Padding fill mode (e.g., mean/reflection).
  - Impact: Avoids artificial borders (black padding is usually worse).
  - Suggested: `mean` is OK; prefer `reflect` if supported.

- `augment.color_jitter`, `augment.color_jitter_p`
  - What: Random brightness/contrast/saturation/hue shifts.
  - Impact: Robust to lighting/white-balance changes.
  - Suggested: `[0.2, 0.2, 0.2, 0.05]` + `p=0.7` (keep). Optionally `p=0.8` if lighting varies a lot.

- `augment.affine.degrees`, `translate`, `scale`, `shear`, `p`
  - What: Rotation/shift/zoom/shear transforms.
  - Impact: Robust to camera angle and framing; usually low risk.
  - Suggested: `degrees=10`, `translate=0.05`, `scale=[0.9,1.1]`, `p=0.3`. If needed, increase `p` to `0.4–0.5`.

- `augment.perspective.distortion`, `augment.perspective.p`
  - What: Perspective warp (near/far distortion).
  - Impact: Helps side-views; too strong can hurt glass/transparent items.
  - Suggested: `distortion=0.15`, `p=0.15` (keep).

- `augment.blur.enabled`, `kernel`, `sigma`, `p`
  - What: Mild blur to simulate defocus/motion.
  - Impact: Helps if real photos are often blurry; can hurt edge-based classes if too strong.
  - Suggested: Keep `false` unless blur is common. If enabling: `p=0.05–0.15`, `sigma=[0.1,1.0]`.

- `augment.random_erasing_p`, `random_erasing_scale`, `random_erasing_ratio`
  - What: Randomly erase a patch (simulate occlusion/reflection/blocked areas).
  - Impact: Improves occlusion robustness; too strong reduces useful signal.
  - Suggested: `p=0.25`, `scale=[0.02,0.12]`, `ratio=[0.3,3.3]` (keep).

- `augment.hflip_p`
  - What: Horizontal flip probability.
  - Impact: Usually safe for material categories.
  - Suggested: `0.5` (keep).

### MixUp / CutMix
- `mixup_cutmix.mixup_alpha`, `mixup_p`
  - What: MixUp strength/prob (image blending).
  - Impact: Smoother decision boundaries; less over-confidence.
  - Suggested: `alpha=0.2`, `p=0.2` (keep).

- `mixup_cutmix.cutmix_alpha`, `cutmix_p`
  - What: CutMix strength/prob (patch replacement).
  - Impact: Reduces background overfitting; helps clutter scenes.
  - Suggested: `alpha=1.0` OK; `p=0.4` is medium. If “clean” accuracy drops, reduce to `0.2–0.3` or weaken in Phase B (you do `0.2`).

### Training Phases
- `phase_a.epochs`, `phase_a.lr`
  - What: Main training stage on the mixed dataset.
  - Impact: Sets baseline representation.
  - Suggested: `epochs=15` OK. `lr=1e-3` is typical for fine-tuning small backbones; if unstable, try `5e-4`.

- `phase_b.epochs`, `phase_b.lr`
  - What: Domain fine-tune stage on in-domain photos.
  - Impact: Biggest driver for “real-world” accuracy.
  - Suggested: `epochs=5–15` . `lr=2e-4` is reasonable; if overfitting, drop to `1e-4`.

- `phase_b.val_ratio`
  - What: Hold-out ratio for domain validation.
  - Impact: Prevents overfitting to small domain set.
  - Suggested: `0.2` (keep).

- `phase_b.prefer_domain_val`
  - What: Select best checkpoint using domain validation metrics.
  - Impact: Prioritizes real-world performance.
  - Suggested: `true` (keep).

- `phase_b.rrc_p`
  - What: Reduce crop aggressiveness during domain fine-tune.
  - Impact: Helps the model “fit” the target distribution.
  - Suggested: `0.7` (keep; could go `0.5` if domain set is small).

- `phase_b.mixup_cutmix.*`
  - What: Weaken MixUp/CutMix during domain fine-tune.
  - Impact: Better convergence to real photos while keeping some regularization.
  - Suggested: `cutmix_p=0.2`, lower mixup alpha.

## Weights & Biases Tracking (Optional)
If you want to monitor training from a browser:
1) Install: `pip install wandb`
2) Login once: `wandb login`
3) Keep `wandb.enabled: true` in `configs/train_v2.yaml`.
