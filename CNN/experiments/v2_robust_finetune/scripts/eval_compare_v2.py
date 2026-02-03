"""Compare baseline vs robust v2 on in-domain data."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
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


@dataclass
class EvalGroup:
    name: str
    items: List[Tuple[Path, int]]


def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config must be a YAML mapping")
    return data


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def list_images_from_folder(
    data_dir: Path,
    label_to_idx: Dict[str, int],
) -> List[Tuple[Path, int]]:
    items: List[Tuple[Path, int]] = []
    for label_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        label = label_dir.name
        if label not in label_to_idx:
            continue
        for file in sorted(label_dir.rglob("*")):
            if file.is_file() and file.suffix.lower() in IMAGE_EXTS:
                items.append((file, label_to_idx[label]))
    return items


def list_images_from_manifest(
    manifest: Path,
    label_to_idx: Dict[str, int],
) -> List[Tuple[Path, int]]:
    items: List[Tuple[Path, int]] = []
    with manifest.open("r", encoding="utf-8") as f:
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
            raise ValueError("Manifest must include label/class/final_label column")
        for row in reader:
            raw_path = row[fieldnames["filepath"]]
            label = row[label_field]
            if label not in label_to_idx:
                continue
            path = Path(raw_path)
            if not path.is_absolute():
                path = manifest.parent / path
            items.append((path, label_to_idx[label]))
    return items


def list_images_from_list(
    list_path: Path,
    label_to_idx: Dict[str, int],
) -> List[Tuple[Path, int]]:
    items: List[Tuple[Path, int]] = []
    with list_path.open("r", encoding="utf-8") as f:
        for line in f:
            rel_path = line.strip()
            if not rel_path:
                continue
            path = list_path.parent / rel_path
            label = path.parent.name
            if label not in label_to_idx:
                continue
            items.append((path, label_to_idx[label]))
    return items


def build_groups(data_dir: Path, label_to_idx: Dict[str, int]) -> List[EvalGroup]:
    clean_dir = data_dir / "clean"
    complex_dir = data_dir / "complex"
    if clean_dir.exists() and complex_dir.exists():
        return [
            EvalGroup("clean", list_images_from_folder(clean_dir, label_to_idx)),
            EvalGroup("complex", list_images_from_folder(complex_dir, label_to_idx)),
        ]

    clean_list = data_dir / "clean_list.txt"
    complex_list = data_dir / "complex_list.txt"
    if clean_list.exists() and complex_list.exists():
        return [
            EvalGroup("clean", list_images_from_list(clean_list, label_to_idx)),
            EvalGroup("complex", list_images_from_list(complex_list, label_to_idx)),
        ]

    manifest_candidates = [data_dir / "manifest.csv", data_dir / "labels.csv"]
    for manifest in manifest_candidates:
        if manifest.exists():
            return [
                EvalGroup(
                    "overall",
                    list_images_from_manifest(manifest, label_to_idx),
                )
            ]

    return [EvalGroup("overall", list_images_from_folder(data_dir, label_to_idx))]


def build_transform(img_size: int, mean: Sequence[float], std: Sequence[float]):
    return transforms.Compose(
        [
            transforms.Resize(
                int(img_size * 1.10),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def run_onnx(
    session: ort.InferenceSession,
    tensor: np.ndarray,
) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    outputs = session.run([output_name], {input_name: tensor.astype(np.float32)})
    return outputs[0]


def evaluate_group(
    session: ort.InferenceSession,
    transform,
    items: List[Tuple[Path, int]],
    num_classes: int,
) -> Tuple[Dict[str, float], List[int], List[int], List[np.ndarray]]:
    total = 0
    top1_correct = 0
    top3_correct = 0
    all_labels: List[int] = []
    all_preds: List[int] = []
    probs_list: List[np.ndarray] = []

    for path, label in tqdm(items, desc="eval", ncols=100):
        with Image.open(path) as opened:
            img = opened.convert("RGB")
        tensor = transform(img).unsqueeze(0).numpy().astype(np.float32)
        logits = run_onnx(session, tensor)
        probs = softmax(logits)
        preds = int(np.argmax(probs))

        total += 1
        top1_correct += int(preds == label)
        top3 = np.argsort(probs)[::-1][: min(3, num_classes)]
        top3_correct += int(label in top3)

        all_labels.append(label)
        all_preds.append(preds)
        probs_list.append(probs)

    if total == 0:
        return {"top1": 0.0, "top3": 0.0, "macro_f1": 0.0}, [], [], []

    top1 = top1_correct / total
    top3 = top3_correct / total
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return (
        {"top1": top1, "top3": top3, "macro_f1": macro_f1},
        all_labels,
        all_preds,
        probs_list,
    )


def save_confusion(
    labels: List[int],
    preds: List[int],
    class_names: List[str],
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
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    except Exception:
        pass


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits).squeeze()
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    return exp / np.sum(exp)


def load_decision_utils():
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from CNN.shared import decision as decision_utils

    return decision_utils


def selective_metrics(
    probs_list: List[np.ndarray],
    labels: List[int],
    class_names: List[str],
    thresholds: dict,
) -> Dict[str, float]:
    decision_utils = load_decision_utils()
    label_names = class_names

    preds = []
    escalated = []
    for probs in probs_list:
        result = decision_utils.decide_from_probs(probs, label_names, thresholds)
        preds.append(label_names.index(result["final_label"]))
        escalated.append(bool(result["escalate"]))

    total = max(1, len(labels))
    non_reject = [i for i, e in enumerate(escalated) if not e]
    coverage = len(non_reject) / total
    if non_reject:
        correct = sum(int(preds[i] == labels[i]) for i in non_reject)
        selective_acc = correct / len(non_reject)
    else:
        selective_acc = 0.0
    return {"coverage": coverage, "selective_acc": selective_acc}


def sweep_thresholds(
    probs_list: List[np.ndarray],
    labels: List[int],
    class_names: List[str],
    base_thresholds: dict,
) -> Dict[str, float]:
    decision_utils = load_decision_utils()
    conf_values = [round(x, 2) for x in np.arange(0.50, 0.951, 0.05)]
    margin_values = [round(x, 2) for x in np.arange(0.05, 0.301, 0.05)]
    best = None

    for conf in conf_values:
        for margin in margin_values:
            thresholds = dict(base_thresholds)
            thresholds["conf"] = conf
            thresholds["margin"] = margin
            preds = []
            escalated = []
            for probs in probs_list:
                result = decision_utils.decide_from_probs(
                    probs,
                    class_names,
                    thresholds,
                )
                preds.append(class_names.index(result["final_label"]))
                escalated.append(bool(result["escalate"]))

            total = max(1, len(labels))
            keep = [i for i, e in enumerate(escalated) if not e]
            coverage = len(keep) / total
            if keep:
                correct = sum(int(preds[i] == labels[i]) for i in keep)
                selective_acc = correct / len(keep)
            else:
                selective_acc = 0.0

            meets = selective_acc >= 0.95
            candidate = {
                "conf": conf,
                "margin": margin,
                "coverage": coverage,
                "selective_acc": selective_acc,
            }
            if meets:
                if best is None or candidate["coverage"] > best["coverage"]:
                    best = candidate
                elif best and candidate["coverage"] == best["coverage"]:
                    if candidate["selective_acc"] > best["selective_acc"]:
                        best = candidate

    if best is None and conf_values and margin_values:
        best = {
            "conf": conf_values[0],
            "margin": margin_values[0],
            "coverage": 0.0,
            "selective_acc": 0.0,
        }
    return best or {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Eval baseline vs robust v2")
    p.add_argument(
        "--baseline-model",
        type=Path,
        default=Path("CNN/models/tier1.onnx"),
    )
    p.add_argument(
        "--baseline-infer",
        type=Path,
        default=Path("CNN/models/infer_config.json"),
    )
    p.add_argument(
        "--robust-model",
        type=Path,
        default=None,
    )
    p.add_argument(
        "--robust-infer",
        type=Path,
        default=None,
    )
    p.add_argument(
        "--label-map",
        type=Path,
        default=Path("CNN/models/label_map.json"),
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("CNN/data/hardset"),
    )
    p.add_argument(
        "--reject-config",
        type=Path,
        default=Path("CNN/experiments/v1_multicrop_reject/configs/infer_v1.yaml"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("CNN/experiments/v2_robust_finetune/outputs"),
    )
    p.add_argument("--prefer-cuda", action="store_true", default=False)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    label_map = load_json(args.label_map)
    class_names = label_map.get("labels", [])
    if not class_names:
        raise ValueError("label_map.json must contain labels")
    label_to_idx = {name: i for i, name in enumerate(class_names)}

    data_dir = args.data_dir
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing data dir: {data_dir}")

    groups = build_groups(data_dir, label_to_idx)
    if not groups:
        raise ValueError("No evaluation data found")

    output_dir = args.output_dir / datetime.now().strftime(
        "eval_compare_v1_%Y%m%d_%H%M%S"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    providers = ["CPUExecutionProvider"]
    if args.prefer_cuda and "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    baseline_infer = load_json(args.baseline_infer)
    b_transform = build_transform(
        int(baseline_infer.get("img_size", 224)),
        baseline_infer.get("mean", [0.485, 0.456, 0.406]),
        baseline_infer.get("std", [0.229, 0.224, 0.225]),
    )

    baseline_session = ort.InferenceSession(
        str(args.baseline_model),
        providers=providers,
    )

    robust_model = args.robust_model
    if robust_model is None:
        last_run = args.output_dir / "last_run.txt"
        if last_run.exists():
            robust_model = Path(
                last_run.read_text(encoding="utf-8").strip()
            ) / "model.onnx"
        else:
            raise FileNotFoundError(
                "Provide --robust-model or ensure last_run.txt exists"
            )

    robust_infer = args.robust_infer
    if robust_infer is None:
        robust_infer = robust_model.parent / "infer_config.json"

    r_infer = load_json(robust_infer)
    r_transform = build_transform(
        int(r_infer.get("img_size", 224)),
        r_infer.get("mean", [0.485, 0.456, 0.406]),
        r_infer.get("std", [0.229, 0.224, 0.225]),
    )
    robust_session = ort.InferenceSession(str(robust_model), providers=providers)

    results: Dict[str, dict] = {"baseline": {}, "robust": {}}
    csv_rows: List[dict] = []

    for group in groups:
        b_metrics, b_labels, b_preds, b_probs = evaluate_group(
            baseline_session,
            b_transform,
            group.items,
            len(class_names),
        )
        r_metrics, r_labels, r_preds, r_probs = evaluate_group(
            robust_session,
            r_transform,
            group.items,
            len(class_names),
        )

        results["baseline"][group.name] = b_metrics
        results["robust"][group.name] = r_metrics

        save_confusion(
            b_labels,
            b_preds,
            class_names,
            output_dir / f"baseline_{group.name}_confusion.png",
        )
        save_confusion(
            r_labels,
            r_preds,
            class_names,
            output_dir / f"robust_{group.name}_confusion.png",
        )

        csv_rows.append(
            {
                "model": "baseline",
                "group": group.name,
                "top1": b_metrics["top1"],
                "top3": b_metrics["top3"],
                "macro_f1": b_metrics["macro_f1"],
            }
        )
        csv_rows.append(
            {
                "model": "robust",
                "group": group.name,
                "top1": r_metrics["top1"],
                "top3": r_metrics["top3"],
                "macro_f1": r_metrics["macro_f1"],
            }
        )

        reject_cfg = load_yaml(args.reject_config)
        thresholds = reject_cfg.get("thresholds", {}) if reject_cfg else {}
        if thresholds:
            thresholds["topk"] = int(reject_cfg.get("output", {}).get("topk", 3))
            results["baseline"][group.name]["selective"] = selective_metrics(
                b_probs,
                b_labels,
                class_names,
                thresholds,
            )
            results["robust"][group.name]["selective"] = selective_metrics(
                r_probs,
                r_labels,
                class_names,
                thresholds,
            )
            results["baseline"][group.name]["sweep_best"] = sweep_thresholds(
                b_probs,
                b_labels,
                class_names,
                thresholds,
            )
            results["robust"][group.name]["sweep_best"] = sweep_thresholds(
                r_probs,
                r_labels,
                class_names,
                thresholds,
            )

    overall_name = "overall" if "overall" in results["baseline"] else groups[0].name
    base_overall = results["baseline"][overall_name]["top1"]
    robust_overall = results["robust"][overall_name]["top1"]
    if robust_overall <= base_overall:
        print(
            "No improvement detected on overall top1. Check augmentations "
            "(scale-down pad, mixup/cutmix), domain fine-tune LR, and label mapping."
        )

    save_json(output_dir / "eval_compare_v1.json", results)

    with (output_dir / "eval_compare_v1.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "group", "top1", "top3", "macro_f1"],
        )
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    print(f"Saved -> {output_dir / 'eval_compare_v1.json'}")
    for row in csv_rows:
        print(
            f"{row['model']} {row['group']} "
            f"top1={row['top1']:.4f} "
            f"top3={row['top3']:.4f} "
            f"f1={row['macro_f1']:.4f}"
        )


if __name__ == "__main__":
    main()
