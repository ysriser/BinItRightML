"""Evaluate v3 model on hardset with temperature calibration and reject-threshold sweep."""

from __future__ import annotations

import argparse
import csv
import json
import logging

import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import onnxruntime as ort
import yaml
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except Exception:
    pillow_heif = None

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
LOGGER = logging.getLogger(__name__)


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config must be a YAML mapping")
    return data


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _list_items_from_manifest(
    manifest: Path,
    label_to_idx: Dict[str, int],
) -> List[Tuple[Path, int]]:
    items: List[Tuple[Path, int]] = []
    with manifest.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row.get("label") or row.get("class") or row.get("final_label")
            if label not in label_to_idx:
                continue
            raw_path = row.get("filepath")
            if not raw_path:
                continue
            path = Path(raw_path)
            if not path.is_absolute():
                path = manifest.parent / path
            if path.exists():
                items.append((path, label_to_idx[label]))
    return items


def _list_items_from_class_dirs(
    data_dir: Path,
    label_to_idx: Dict[str, int],
) -> List[Tuple[Path, int]]:
    items: List[Tuple[Path, int]] = []
    for class_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        if class_dir.name not in label_to_idx:
            continue
        idx = label_to_idx[class_dir.name]
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTS:
                items.append((image_path, idx))
    return items


def list_items(data_dir: Path, label_to_idx: Dict[str, int]) -> List[Tuple[Path, int]]:
    manifest = data_dir / "manifest.csv"
    if manifest.exists():
        return _list_items_from_manifest(manifest, label_to_idx)
    return _list_items_from_class_dirs(data_dir, label_to_idx)


def stratified_split(
    items: List[Tuple[Path, int]],
    ratio: float,
    seed: int,
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    by_class: Dict[int, List[Tuple[Path, int]]] = defaultdict(list)
    for item in items:
        by_class[item[1]].append(item)

    rng = random.Random(seed)
    calib: List[Tuple[Path, int]] = []
    eval_items: List[Tuple[Path, int]] = []

    for class_items in by_class.values():
        class_copy = class_items[:]
        rng.shuffle(class_copy)
        split_n = max(1, int(round(len(class_copy) * ratio))) if len(class_copy) > 1 else 1
        split_n = min(split_n, len(class_copy) - 1) if len(class_copy) > 1 else 1
        calib.extend(class_copy[:split_n])
        eval_items.extend(class_copy[split_n:])

    if not eval_items:
        eval_items = calib[:]
    return calib, eval_items


def build_transform(img_size: int, mean: Sequence[float], std: Sequence[float]):
    return transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.15), interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def run_logits(
    session: ort.InferenceSession,
    transform,
    items: List[Tuple[Path, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    logits_list: List[np.ndarray] = []
    labels_list: List[int] = []

    for image_path, label_idx in tqdm(items, desc="inference", ncols=100):
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        tensor = transform(image).unsqueeze(0).numpy().astype(np.float32)
        logits = session.run([output_name], {input_name: tensor})[0]
        logits_list.append(np.asarray(logits).reshape(-1))
        labels_list.append(label_idx)

    if not logits_list:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    return np.stack(logits_list).astype(np.float32), np.asarray(labels_list, dtype=np.int64)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def negative_log_likelihood(probs: np.ndarray, labels: np.ndarray) -> float:
    eps = 1e-9
    idx = np.arange(labels.shape[0])
    return float(-np.mean(np.log(np.clip(probs[idx, labels], eps, 1.0))))


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    t_min: float,
    t_max: float,
    t_step: float,
) -> Tuple[float, float]:
    best_t = 1.0
    best_nll = float("inf")
    current = t_min
    while current <= t_max + 1e-9:
        probs = softmax(logits / current)
        nll = negative_log_likelihood(probs, labels)
        if nll < best_nll:
            best_nll = nll
            best_t = current
        current += t_step
    return round(best_t, 4), best_nll


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    bins: int,
) -> float:
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correctness = (predictions == labels).astype(np.float32)

    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    total = len(labels)
    for i in range(bins):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]
        if i == bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        if not np.any(mask):
            continue
        acc = float(np.mean(correctness[mask]))
        conf = float(np.mean(confidences[mask]))
        ece += (np.sum(mask) / total) * abs(acc - conf)
    return float(ece)


def metric_summary(probs: np.ndarray, labels: np.ndarray) -> dict:
    preds = np.argmax(probs, axis=1)
    top1 = float(np.mean(preds == labels)) if len(labels) else 0.0

    topk = min(3, probs.shape[1]) if probs.size else 0
    topk_correct = 0
    if topk > 0:
        for i in range(len(labels)):
            topk_idx = np.argsort(probs[i])[::-1][:topk]
            topk_correct += int(labels[i] in topk_idx)
    top3 = float(topk_correct / len(labels)) if len(labels) else 0.0

    macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0)) if len(labels) else 0.0
    nll = negative_log_likelihood(probs, labels) if len(labels) else 0.0
    return {
        "top1": top1,
        "top3": top3,
        "macro_f1": macro_f1,
        "nll": nll,
    }


def apply_reject_rules(
    probs: np.ndarray,
    class_names: Sequence[str],
    conf: float,
    margin: float,
    strict_per_class: Dict[str, float],
    reject_label: str = "other_uncertain",
) -> Tuple[np.ndarray, np.ndarray]:
    preds = np.argmax(probs, axis=1)
    sorted_probs = np.sort(probs, axis=1)[:, ::-1]
    top1 = sorted_probs[:, 0]
    top2 = sorted_probs[:, 1] if probs.shape[1] > 1 else np.zeros_like(top1)

    reject_idx = class_names.index(reject_label) if reject_label in class_names else None

    final_preds = preds.copy()
    escalated = np.zeros_like(preds, dtype=bool)
    for i, pred_idx in enumerate(preds):
        label_name = class_names[pred_idx]
        label_conf_threshold = max(conf, float(strict_per_class.get(label_name, conf)))
        should_escalate = (
            label_name == reject_label
            or top1[i] < label_conf_threshold
            or (top1[i] - top2[i]) < margin
        )
        escalated[i] = should_escalate
        if should_escalate and reject_idx is not None:
            final_preds[i] = reject_idx
    return final_preds, escalated


def _build_sweep_row(
    probs: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    strict_per_class: Dict[str, float],
    conf: float,
    margin: float,
) -> dict:
    pred_after, escalated = apply_reject_rules(
        probs=probs,
        class_names=class_names,
        conf=float(round(conf, 4)),
        margin=float(round(margin, 4)),
        strict_per_class=strict_per_class,
    )
    keep_idx = np.nonzero(~escalated)[0]
    coverage = float(len(keep_idx) / len(labels)) if len(labels) else 0.0
    selective_acc = (
        float(np.mean(pred_after[keep_idx] == labels[keep_idx])) if len(keep_idx) else 0.0
    )
    overall_top1 = float(np.mean(pred_after == labels)) if len(labels) else 0.0
    return {
        "conf": round(float(conf), 4),
        "margin": round(float(margin), 4),
        "coverage": round(coverage, 6),
        "selective_acc": round(selective_acc, 6),
        "overall_top1_after_reject": round(overall_top1, 6),
    }


def _is_better_sweep_candidate(candidate: dict, current_best: dict | None) -> bool:
    if current_best is None:
        return True
    if candidate["coverage"] > current_best["coverage"]:
        return True
    return (
        candidate["coverage"] == current_best["coverage"]
        and candidate["selective_acc"] > current_best["selective_acc"]
    )


def sweep_thresholds(
    probs: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    sweep_cfg: dict,
) -> Tuple[dict, List[dict]]:
    conf_values = np.arange(
        float(sweep_cfg.get("conf_min", 0.5)),
        float(sweep_cfg.get("conf_max", 0.95)) + 1e-9,
        float(sweep_cfg.get("conf_step", 0.05)),
    )
    margin_values = np.arange(
        float(sweep_cfg.get("margin_min", 0.03)),
        float(sweep_cfg.get("margin_max", 0.30)) + 1e-9,
        float(sweep_cfg.get("margin_step", 0.03)),
    )
    
    strict_per_class = sweep_cfg.get("strict_per_class", {}) or {}
    target = float(sweep_cfg.get("target_selective_acc", 0.95))

    rows: List[dict] = []
    best: dict | None = None

    for conf in conf_values:
        for margin in margin_values:
            row = _build_sweep_row(
                probs=probs,
                labels=labels,
                class_names=class_names,
                strict_per_class=strict_per_class,
                conf=float(conf),
                margin=float(margin),
            )
            rows.append(row)

            if row["selective_acc"] >= target and _is_better_sweep_candidate(row, best):
                best = row

    if best is None and rows:
        rows_sorted = sorted(rows, key=lambda x: (x["selective_acc"], x["coverage"]), reverse=True)
        best = rows_sorted[0]
    return best or {}, rows


def save_confusion_image(
    labels: np.ndarray,
    preds: np.ndarray,
    class_names: Sequence[str],
    out_path: Path,
) -> None:
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
    except Exception as exc:
        LOGGER.warning(
            "Failed to render confusion matrix at %s: %s",
            out_path,
            exc,
        )


def save_reliability_plot(
    probs_before: np.ndarray,
    probs_after: np.ndarray,
    labels: np.ndarray,
    bins: int,
    out_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt

        def _curve(probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            conf = np.max(probs, axis=1)
            pred = np.argmax(probs, axis=1)
            corr = (pred == labels).astype(np.float32)
            edges = np.linspace(0.0, 1.0, bins + 1)
            x_vals = []
            y_vals = []
            for i in range(bins):
                lo, hi = edges[i], edges[i + 1]
                mask = (conf >= lo) & (conf <= hi) if i == bins - 1 else (conf >= lo) & (conf < hi)
                if np.any(mask):
                    x_vals.append(float(np.mean(conf[mask])))
                    y_vals.append(float(np.mean(corr[mask])))
            return np.array(x_vals), np.array(y_vals)

        xb, yb = _curve(probs_before)
        xa, ya = _curve(probs_after)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], "k--", label="Perfect")
        if xb.size:
            ax.plot(xb, yb, marker="o", label="Before calibration")
        if xa.size:
            ax.plot(xa, ya, marker="o", label="After calibration")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title("Reliability Diagram")
        ax.legend(loc="best")
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
    except Exception as exc:
        LOGGER.warning(
            "Failed to render reliability plot at %s: %s",
            out_path,
            exc,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate calibration for v3 model")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("CNN/experiments/v3_generalization_upgrade/configs/eval_v3.yaml"),
    )
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--infer", type=Path, default=None)
    parser.add_argument("--label-map", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--prefer-cuda", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ml_root = Path(__file__).resolve().parents[4]

    config_path = args.config
    if not config_path.is_absolute():
        config_path = ml_root / config_path
    cfg = load_yaml(config_path)

    paths = cfg.get("paths", {})

    model_path = args.model or Path(paths.get("robust_model", "CNN/models/tier1.onnx"))
    infer_path = args.infer or Path(paths.get("robust_infer", "CNN/models/infer_config.json"))
    label_map_path = args.label_map or Path(paths.get("label_map", "CNN/models/label_map.json"))
    data_dir = args.data_dir or Path(paths.get("hardset_dir", "CNN/data/hardset"))
    output_root = args.output_dir or Path(paths.get("output_dir", "CNN/experiments/v3_generalization_upgrade/outputs"))

    if not model_path.is_absolute():
        model_path = ml_root / model_path
    if not infer_path.is_absolute():
        infer_path = ml_root / infer_path
    if not label_map_path.is_absolute():
        label_map_path = ml_root / label_map_path
    if not data_dir.is_absolute():
        data_dir = ml_root / data_dir
    if not output_root.is_absolute():
        output_root = ml_root / output_root

    infer_cfg = load_json(infer_path)
    label_map = load_json(label_map_path)
    class_names = label_map.get("labels", [])
    if not class_names:
        raise ValueError("label_map.json must contain labels")
    label_to_idx = {name: i for i, name in enumerate(class_names)}

    items = list_items(data_dir, label_to_idx)
    if not items:
        raise ValueError(f"No images found under {data_dir}")

    calib_cfg = cfg.get("calibration", {}) or {}
    calib_ratio = float(calib_cfg.get("split_ratio", 0.4))
    seed = int(calib_cfg.get("seed", 42))
    calib_items, eval_items = stratified_split(items, calib_ratio, seed)

    providers = ["CPUExecutionProvider"]
    if args.prefer_cuda and "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    session = ort.InferenceSession(str(model_path), providers=providers)
    transform = build_transform(
        int(infer_cfg.get("img_size", 224)),
        infer_cfg.get("mean", [0.485, 0.456, 0.406]),
        infer_cfg.get("std", [0.229, 0.224, 0.225]),
    )

    calib_logits, calib_labels = run_logits(session, transform, calib_items)
    eval_logits, eval_labels = run_logits(session, transform, eval_items)

    if calib_logits.size == 0 or eval_logits.size == 0:
        raise ValueError("Calibration or evaluation split is empty")

    probs_eval_raw = softmax(eval_logits)

    t_cfg = calib_cfg.get("temperature_search", {}) or {}
    t_min = float(t_cfg.get("min", 0.7))
    t_max = float(t_cfg.get("max", 3.0))
    t_step = float(t_cfg.get("step", 0.05))
    temperature, calib_nll = fit_temperature(calib_logits, calib_labels, t_min, t_max, t_step)

    probs_eval_cal = softmax(eval_logits / temperature)

    ece_bins = int((cfg.get("ece", {}) or {}).get("bins", 15))
    raw_metrics = metric_summary(probs_eval_raw, eval_labels)
    cal_metrics = metric_summary(probs_eval_cal, eval_labels)
    raw_metrics["ece"] = expected_calibration_error(probs_eval_raw, eval_labels, ece_bins)
    cal_metrics["ece"] = expected_calibration_error(probs_eval_cal, eval_labels, ece_bins)

    sweep_cfg = cfg.get("reject_sweep", {}) or {}
    best_thresholds, sweep_rows = sweep_thresholds(
        probs=probs_eval_cal,
        labels=eval_labels,
        class_names=class_names,
        sweep_cfg=sweep_cfg,
    )

    run_name = datetime.now().strftime("v3_eval_%Y%m%d_%H%M%S")
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_preds = np.argmax(probs_eval_raw, axis=1)
    cal_preds = np.argmax(probs_eval_cal, axis=1)
    save_confusion_image(eval_labels, raw_preds, class_names, run_dir / "confusion_raw.png")
    save_confusion_image(eval_labels, cal_preds, class_names, run_dir / "confusion_calibrated.png")
    save_reliability_plot(
        probs_before=probs_eval_raw,
        probs_after=probs_eval_cal,
        labels=eval_labels,
        bins=ece_bins,
        out_path=run_dir / "reliability_before_after.png",
    )

    sweep_csv = run_dir / "threshold_sweep.csv"
    with sweep_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["conf", "margin", "coverage", "selective_acc", "overall_top1_after_reject"],
        )
        writer.writeheader()
        for row in sweep_rows:
            writer.writerow(row)

    recommended = {
        "conf_threshold": best_thresholds.get("conf"),
        "margin_threshold": best_thresholds.get("margin"),
        "strict_per_class": sweep_cfg.get("strict_per_class", {}),
        "coverage": best_thresholds.get("coverage", 0.0),
        "selective_acc": best_thresholds.get("selective_acc", 0.0),
    }

    summary = {
        "model": str(model_path),
        "infer_config": str(infer_path),
        "label_map": str(label_map_path),
        "data_dir": str(data_dir),
        "providers": providers,
        "splits": {
            "calibration_count": len(calib_items),
            "evaluation_count": len(eval_items),
        },
        "temperature_scaling": {
            "temperature": temperature,
            "calibration_nll": calib_nll,
        },
        "metrics_raw": raw_metrics,
        "metrics_calibrated": cal_metrics,
        "recommended_reject_thresholds": recommended,
    }

    save_json(run_dir / "eval_v3_calibration.json", summary)

    print(f"Saved calibration report -> {run_dir / 'eval_v3_calibration.json'}")
    print(f"Saved threshold sweep -> {sweep_csv}")
    print(
        "Recommended thresholds: "
        f"conf={recommended['conf_threshold']} margin={recommended['margin_threshold']} "
        f"coverage={recommended['coverage']:.4f} selective_acc={recommended['selective_acc']:.4f}"
    )


if __name__ == "__main__":
    main()
