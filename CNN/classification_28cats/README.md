# Bin-It-Right ML (TACO Super-28 Classification)

This module builds a **classification** dataset from TACO (COCO format), trains a model on **28 super-categories**, evaluates, exports, and serves a FastAPI inference API.

## Layout
```
ml/
  classification_28cats/
    src/
      data/
        taco_coco.py
        transforms.py
        splits.py
        adapters.py
      models/
        backbone.py
        classifier.py
      rules.py
      router.py
      utils.py
    configs/
      taco_super28.yaml
      label_map_taco_super28.yaml
    train.py
    eval.py
    export.py
    serve.py
    tests/
  data_tools/
    build_taco_cls_dataset.py
  requirements_ml.txt
  README.md
```

## 1) Build dataset (COCO -> classification crops)
```bash
python ml/classification_28cats/data_tools/build_taco_cls_dataset.py \
  --images_dir <path_to_images> \
  --coco_json <path_to_coco_annotations.json> \
  --out_dir ml/data/taco_super28_cls \
  --split_ratios 0.8 0.1 0.1 \
  --seed 42 \
  --pad_ratio 0.10 \
  --min_box_area_ratio 0.01
```

### Example (TACO official)
```bash
python ml/classification_28cats/data_tools/build_taco_cls_dataset.py \
  --images_dir TACO/data \
  --coco_json TACO/data/annotations.json \
  --out_dir ml/data/taco_super28_cls
```

### Example (TACO unofficial)
```bash
python ml/classification_28cats/data_tools/build_taco_cls_dataset.py \
  --images_dir TACO/data \
  --coco_json TACO/data/annotations_unofficial.json \
  --out_dir ml/data/taco_super28_cls_unofficial
```

### Example (combine official + unofficial)
```bash
python ml/classification_28cats/data_tools/build_taco_cls_dataset.py \
  --images_dir TACO/data \
  --coco_json TACO/data/annotations.json TACO/data/annotations_unofficial.json \
  --out_dir ml/data/taco_super28_cls_combined \
  --label_map ml/classification_28cats/configs/label_map_taco_super28_combined.yaml
```
There is also a compatibility wrapper:
```bash
python ml/data_tools/build_taco_cls_dataset.py ...
```
This creates:
```
ml/data/taco_super28_cls/{train|val|test}/{super_class}/<imageid>_<annid>.jpg
```
It also **auto-generates** `ml/classification_28cats/configs/label_map_taco_super28.yaml`.

## 2) Train
Edit config (choose a backbone):
```
ml/classification_28cats/configs/taco_super28.yaml
```
Supported backbones:
- `efficientnet_b0`
- `efficientnet_b3`
- `mobilenet_v3_large`
- `convnext_tiny`
Run:
```bash
python ml/classification_28cats/train.py
```

Outputs:
- `ml/outputs/classification_28cats/<run_name>/` (metrics, plots, reports)
- `ml/artifacts/classification_28cats/<run_name>/best.pt`

## 3) Evaluate
```bash
python ml/classification_28cats/eval.py --checkpoint ml/artifacts/classification_28cats/<run_name>/best.pt
```

## 4) Export
```bash
python ml/classification_28cats/export.py --checkpoint ml/artifacts/classification_28cats/<run_name>/best.pt
```
Exports to:
```
ml/artifacts/classification_28cats/<run_name>/model.ts
ml/artifacts/classification_28cats/latest/model.ts
```

## 5) Serve
```bash
python ml/classification_28cats/serve.py
```
Endpoints:
- `GET /health`
- `POST /api/v1/scan` (multipart field name: `image`)

## Install
```bash
pip install -r ml/requirements.txt
```

## Notes
- GPU is default in config (`device: cuda`, `require_cuda: true`).
- Rules engine lives in `ml/classification_28cats/src/rules.py`.
- Router stub (Phase 2 hook) in `ml/classification_28cats/src/router.py`.
