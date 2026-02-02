"""Evaluate V1 multicrop + reject on G3_SGData."""

from __future__ import annotations

import sys

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import yaml
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CNN.shared import decision, multicrop, onnx_infer, preprocess

IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".heic",
    ".heif",
}


def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML mapping: {path}")
    return data


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def merge_paths(base: dict, override: dict) -> dict:
    for key in ("models", "data", "outputs"):
        if key in override:
            base.setdefault(key, {})
            base[key].update(override[key] or {})
    return base


def resolve_path(path_str: str, repo_root: Path) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else repo_root / path


def list_images_from_folders(
    data_dir: Path,
    labels: Sequence[str],
) -> List[Tuple[Path, str]]:
    items: List[Tuple[Path, str]] = []
    label_set = set(labels)
    for label_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        label = label_dir.name
        if label not in label_set:
            continue
        for file in sorted(label_dir.rglob("*")):
            if file.is_file() and file.suffix.lower() in IMAGE_EXTS:
                items.append((file, label))
    return items


def load_manifest(manifest_path: Path, labels: Sequence[str]) -> List[Tuple[Path, str]]:
    label_set = set(labels)
    with manifest_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("Manifest must include headers")
        fieldnames = {name.lower(): name for name in reader.fieldnames}
        if "filepath" not in fieldnames:
            raise ValueError("Manifest must include 'filepath' column")
        label_field = (
            fieldnames.get("label")
            or fieldnames.get("class")
            or fieldnames.get("final_label")
        )
        if not label_field:
            raise ValueError(
                "Manifest must include a label column (label/class/final_label)"
            )

        items: List[Tuple[Path, str]] = []
        for row in reader:
            raw_path = row[fieldnames["filepath"]]
            raw_label = row[label_field]
            if raw_label not in label_set:
                continue
            path = Path(raw_path)
            if not path.is_absolute():
                path = manifest_path.parent / path
            items.append((path, raw_label))
    return items


def resolve_items(data_dir: Path, labels: Sequence[str]) -> List[Tuple[Path, str]]:
    manifest_candidates = [data_dir / "manifest.csv", data_dir / "labels.csv"]
    for candidate in manifest_candidates:
        if candidate.exists():
            return load_manifest(candidate, labels)
    return list_images_from_folders(data_dir, labels)


def evaluate_probs(
    probs_list: Sequence[np.ndarray],
    true_labels: Sequence[str],
    labels: Sequence[str],
    thresholds: Dict[str, Any],
) -> Dict[str, Any]:
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    preds: List[str] = []
    escalated: List[bool] = []

    for probs, true_label in zip(probs_list, true_labels):
        result = decision.decide_from_probs(probs, labels, thresholds)
        preds.append(result["final_label"])
        escalated.append(bool(result["escalate"]))

    y_true = [label_to_idx[label] for label in true_labels]
    y_pred = [label_to_idx[label] for label in preds]

    total = len(y_true)
    top1 = sum(int(p == y) for p, y in zip(y_pred, y_true)) / max(total, 1)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm_all = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))

    non_reject_indices = [idx for idx, esc in enumerate(escalated) if not esc]
    coverage = len(non_reject_indices) / max(total, 1)
    if non_reject_indices:
        y_true_nr = [y_true[idx] for idx in non_reject_indices]
        y_pred_nr = [y_pred[idx] for idx in non_reject_indices]
        correct = sum(int(p == y) for p, y in zip(y_pred_nr, y_true_nr))
        selective_acc = correct / max(len(y_true_nr), 1)
        cm_non = confusion_matrix(y_true_nr, y_pred_nr, labels=list(range(len(labels))))
    else:
        selective_acc = 0.0
        cm_non = None

    return {
        "top1": top1,
        "macro_f1": macro_f1,
        "coverage": coverage,
        "selective_acc": selective_acc,
        "confusion_all": cm_all.tolist(),
        "confusion_non_reject": None if cm_non is None else cm_non.tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V1 multicrop evaluation on G3_SGData"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "CNN/experiments/v1_multicrop_reject/configs/infer_v1.yaml"
        ),
    )
    parser.add_argument(
        "--paths",
        type=Path,
        default=Path("CNN/experiments/v1_multicrop_reject/configs/paths.yaml"),
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--max-images", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[4]
    infer_cfg = load_yaml(resolve_path(str(args.config), repo_root))
    infer_cfg = merge_paths(
        infer_cfg,
        load_yaml(resolve_path(str(args.paths), repo_root)),
    )

    model_paths = infer_cfg.get("models", {})
    onnx_path = resolve_path(
        str(model_paths.get("onnx", "CNN/models/tier1.onnx")),
        repo_root,
    )
    label_map_path = resolve_path(
        str(model_paths.get("label_map", "CNN/models/label_map.json")),
        repo_root,
    )
    infer_json_path = resolve_path(
        str(model_paths.get("infer_config", "CNN/models/infer_config.json")), repo_root
    )

    label_map = load_json(label_map_path)
    labels = preprocess.resolve_labels(label_map)
    preprocess.validate_labels(labels)

    infer_json = load_json(infer_json_path)
    img_size = int(infer_json.get("img_size", 224))
    mean, std = preprocess.resolve_mean_std(infer_json)

    output_cfg = infer_cfg.get("output", {}) or {}
    topk = int(output_cfg.get("topk", infer_json.get("topk", 3)))

    thresholds_v1 = infer_cfg.get("thresholds", {}) or {}
    thresholds_v1["topk"] = topk

    thresholds_baseline = {
        "conf": float(infer_json.get("conf_threshold", 0.75)),
        "margin": float(infer_json.get("margin_threshold", 0.15)),
        "reject_to_other": True,
        "topk": topk,
    }

    multicrop_cfg = infer_cfg.get("multicrop", {}) or {}

    data_dir = args.data_dir
    if data_dir is None:
        data_dir = Path(
            infer_cfg.get("data", {}).get("g3_sgdata", "CNN/data/G3_SGData")
        )
    data_dir = resolve_path(str(data_dir), repo_root)

    items = resolve_items(data_dir, labels)
    if args.max_images and args.max_images > 0:
        items = items[: args.max_images]
    if not items:
        raise ValueError("No images found in data-dir")

    onnx_model = onnx_infer.OnnxInfer(onnx_path)

    def infer_fn(batch: np.ndarray, _: str | None = None) -> np.ndarray:
        return onnx_model.run(batch)

    true_labels: List[str] = []
    probs_baseline: List[np.ndarray] = []
    probs_v1: List[np.ndarray] = []
    skipped: List[str] = []

    for path, label in tqdm(items, desc="Inference", ncols=100):
        try:
            with Image.open(path) as opened:
                img = preprocess.ensure_rgb(opened).copy()
        except Exception:
            skipped.append(str(path))
            continue

        base_tensor = preprocess.preprocess_image(
            img,
            img_size,
            mean,
            std,
            mode="center",
        )
        base_logits = infer_fn(base_tensor)
        base_probs = decision.softmax(base_logits)

        v1_probs, _ = multicrop.run_multicrop(
            img=img,
            infer_fn=infer_fn,
            img_size=img_size,
            mean=mean,
            std=std,
            cfg=multicrop_cfg,
        )

        true_labels.append(label)
        probs_baseline.append(base_probs)
        probs_v1.append(v1_probs)

    output_dir = Path(
        infer_cfg.get("outputs", {}).get(
            "base_dir",
            "CNN/experiments/v1_multicrop_reject/outputs",
        )
    )
    output_dir = resolve_path(str(output_dir), repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_metrics = evaluate_probs(
        probs_baseline,
        true_labels,
        labels,
        thresholds_baseline,
    )
    v1_metrics = evaluate_probs(probs_v1, true_labels, labels, thresholds_v1)

    sweep_rows: List[Dict[str, float]] = []
    conf_values = [round(x, 2) for x in np.arange(0.50, 0.951, 0.05)]
    margin_values = [round(x, 2) for x in np.arange(0.05, 0.301, 0.05)]

    best_choice = None
    for conf_th in conf_values:
        for margin_th in margin_values:
            sweep_thresholds = dict(thresholds_v1)
            sweep_thresholds["conf"] = conf_th
            sweep_thresholds["margin"] = margin_th
            metrics = evaluate_probs(probs_v1, true_labels, labels, sweep_thresholds)
            row = {
                "conf_threshold": conf_th,
                "margin_threshold": margin_th,
                "coverage": metrics["coverage"],
                "selective_acc": metrics["selective_acc"],
            }
            sweep_rows.append(row)

            meets = metrics["selective_acc"] >= 0.95
            if meets:
                if best_choice is None:
                    best_choice = row
                else:
                    if row["coverage"] > best_choice["coverage"]:
                        best_choice = row
                    elif row["coverage"] == best_choice["coverage"] and row[
                        "selective_acc"
                    ] > best_choice["selective_acc"]:
                        best_choice = row

    if best_choice is None and sweep_rows:
        best_choice = max(sweep_rows, key=lambda r: r["selective_acc"])

    output_json = {
        "labels": labels,
        "total": len(true_labels),
        "skipped": len(skipped),
        "baseline": baseline_metrics,
        "v1": v1_metrics,
        "recommended_thresholds": best_choice,
    }

    json_path = output_dir / "eval_custom_v1.json"
    csv_path = output_dir / "eval_custom_v1.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "conf_threshold",
                "margin_threshold",
                "coverage",
                "selective_acc",
            ],
        )
        writer.writeheader()
        for row in sweep_rows:
            writer.writerow(row)

    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")
    if best_choice:
        print(
            "Recommended thresholds -> "
            f"conf={best_choice['conf_threshold']} "
            f"margin={best_choice['margin_threshold']} "
            f"coverage={best_choice['coverage']:.3f} "
            f"selective_acc={best_choice['selective_acc']:.3f}"
        )


if __name__ == "__main__":
    main()
