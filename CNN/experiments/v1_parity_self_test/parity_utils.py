"""Utilities for inference parity self-test."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    import onnxruntime as ort

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


@dataclass(frozen=True)
class InferConfig:
    img_size: int
    mean: List[float]
    std: List[float]
    resize_scale: float
    interpolation: str


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_labels(label_map_path: Path) -> List[str]:
    label_map = load_json(label_map_path)
    labels = label_map.get("labels")
    if not isinstance(labels, list) or not labels:
        raise ValueError("label_map.json must contain a non-empty 'labels' list")
    return [str(label) for label in labels]


def load_infer_config(
    infer_path: Path,
    resize_scale: float = 1.15,
) -> InferConfig:
    cfg = load_json(infer_path)
    img_size = int(cfg.get("img_size", 224))
    mean = [float(x) for x in cfg.get("mean", [0.485, 0.456, 0.406])]
    std = [float(x) for x in cfg.get("std", [0.229, 0.224, 0.225])]
    return InferConfig(
        img_size=img_size,
        mean=mean,
        std=std,
        resize_scale=float(resize_scale),
        interpolation="BICUBIC",
    )


def list_images_from_folder(data_dir: Path) -> List[Path]:
    items: List[Path] = []
    for file in sorted(data_dir.rglob("*")):
        if file.is_file() and file.suffix.lower() in IMAGE_EXTS:
            items.append(file)
    return items


def _list_items_from_manifest(manifest_path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
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
        for row in reader:
            raw_path = row[fieldnames["filepath"]]
            path = Path(raw_path)
            if not path.is_absolute():
                path = manifest_path.parent / path
            label = row[label_field] if label_field else None
            items.append(
                {
                    "path": path,
                    "label": label,
                    "image_id": raw_path,
                }
            )
    return items


def _list_items_from_folder(images_dir: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for path in list_images_from_folder(images_dir):
        image_id = str(path.relative_to(images_dir))
        label = None if path.parent == images_dir else path.parent.name
        items.append({"path": path, "label": label, "image_id": image_id})
    return items


def list_items(
    images_dir: Optional[Path],
    manifest_path: Optional[Path],
) -> List[Dict[str, Any]]:
    if images_dir is None and manifest_path is None:
        raise ValueError("Provide --images or --manifest")

    if manifest_path is not None:
        return _list_items_from_manifest(manifest_path)

    if images_dir is not None:
        return _list_items_from_folder(images_dir)
    return []


def resize_shorter_side(
    img: Image.Image,
    size: int,
    resample: int = Image.BICUBIC,
) -> Image.Image:
    width, height = img.size
    short = min(width, height)
    if short == size:
        return img
    scale = size / float(short)
    new_w = int(round(width * scale))
    new_h = int(round(height * scale))
    return img.resize((new_w, new_h), resample=resample)


def center_crop(
    img: Image.Image,
    size: int,
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    width, height = img.size
    left = max(0, int(round((width - size) / 2.0)))
    top = max(0, int(round((height - size) / 2.0)))
    right = left + size
    bottom = top + size
    return img.crop((left, top, right, bottom)), (left, top, right, bottom)


def to_tensor(
    img: Image.Image,
    mean: Sequence[float],
    std: Sequence[float],
) -> np.ndarray:
    arr = np.asarray(img).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    mean_arr = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std_arr = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    arr = (arr - mean_arr) / std_arr
    arr = np.transpose(arr, (2, 0, 1))
    return arr[None, :, :, :]


def preprocess_with_audit(
    img: Image.Image,
    cfg: InferConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    original_size = img.size
    resized = resize_shorter_side(
        img,
        int(round(cfg.img_size * cfg.resize_scale)),
        resample=Image.BICUBIC,
    )
    resized_size = resized.size
    cropped, crop_box = center_crop(resized, cfg.img_size)
    tensor = to_tensor(cropped, cfg.mean, cfg.std)

    audit = {
        "original_size": list(original_size),
        "resized_size": list(resized_size),
        "crop_box": list(crop_box),
        "channel_order": "RGB",
        "pixel_scaling": "0-1",
        "normalize_mean": cfg.mean,
        "normalize_std": cfg.std,
        "resize_scale": cfg.resize_scale,
        "interpolation": cfg.interpolation,
    }
    return tensor, audit


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits).squeeze()
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    return exp / np.sum(exp)


def topk(probs: np.ndarray, labels: Sequence[str], k: int) -> List[Dict[str, float]]:
    k = max(1, min(k, len(labels)))
    top_idxs = np.argsort(probs)[::-1][:k]
    return [{"label": labels[i], "p": float(probs[i])} for i in top_idxs]


def hash_array(arr: np.ndarray) -> str:
    arr = np.asarray(arr, dtype=np.float32)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def build_session(model_path: Path, prefer_cuda: bool = True) -> "ort.InferenceSession":
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "onnxruntime is required for parity inference. "
            "Install it or skip parity execution in CI."
        ) from exc
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")
    providers = ["CPUExecutionProvider"]
    if prefer_cuda and "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ort.InferenceSession(str(model_path), providers=providers)


def run_onnx(session: "ort.InferenceSession", tensor: np.ndarray) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    outputs = session.run([output_name], {input_name: tensor.astype(np.float32)})
    return outputs[0]


def record_for_image(
    session: ort.InferenceSession,
    cfg: InferConfig,
    labels: Sequence[str],
    path: Path,
    label: Optional[str],
    image_id: str,
    topk_n: int,
) -> Dict[str, Any]:
    with Image.open(path) as opened:
        img = opened.convert("RGB")
    tensor, audit = preprocess_with_audit(img, cfg)
    logits = run_onnx(session, tensor)
    probs = softmax(logits)

    topk_list = topk(probs, labels, topk_n)
    top1_label = topk_list[0]["label"]
    top1_prob = topk_list[0]["p"]

    return {
        "image_id": image_id,
        "filepath": str(path),
        "label": label,
        "top1_label": top1_label,
        "top1_prob": float(top1_prob),
        "top3": topk_list,
        "probs": probs.tolist(),
        "prob_hash": hash_array(probs),
        "input_tensor_hash": hash_array(tensor),
        "audit": audit,
    }


def build_output(
    records: List[Dict[str, Any]],
    model_path: Path,
    infer_path: Path,
    label_map_path: Path,
    cfg: InferConfig,
) -> Dict[str, Any]:
    return {
        "meta": {
            "model_path": str(model_path),
            "infer_config": str(infer_path),
            "label_map": str(label_map_path),
            "img_size": cfg.img_size,
            "mean": cfg.mean,
            "std": cfg.std,
            "resize_scale": cfg.resize_scale,
            "interpolation": cfg.interpolation,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "total": len(records),
        },
        "images": records,
    }


def write_csv(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_id",
        "filepath",
        "label",
        "top1_label",
        "top1_prob",
        "top3_labels",
        "top3_probs",
        "prob_hash",
        "input_tensor_hash",
        "original_size",
        "resized_size",
        "crop_box",
        "resize_scale",
        "interpolation",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            top3_labels = ",".join([x["label"] for x in record["top3"]])
            top3_probs = ",".join([f"{x['p']:.6f}" for x in record["top3"]])
            audit = record.get("audit", {})
            writer.writerow(
                {
                    "image_id": record.get("image_id"),
                    "filepath": record.get("filepath"),
                    "label": record.get("label"),
                    "top1_label": record.get("top1_label"),
                    "top1_prob": record.get("top1_prob"),
                    "top3_labels": top3_labels,
                    "top3_probs": top3_probs,
                    "prob_hash": record.get("prob_hash"),
                    "input_tensor_hash": record.get("input_tensor_hash"),
                    "original_size": audit.get("original_size"),
                    "resized_size": audit.get("resized_size"),
                    "crop_box": audit.get("crop_box"),
                    "resize_scale": audit.get("resize_scale"),
                    "interpolation": audit.get("interpolation"),
                }
            )


def compare_audit(golden: dict, current: dict) -> List[str]:
    reasons: List[str] = []
    fields = [
        "original_size",
        "resized_size",
        "crop_box",
        "channel_order",
        "pixel_scaling",
        "normalize_mean",
        "normalize_std",
        "resize_scale",
        "interpolation",
    ]
    for field in fields:
        if golden.get(field) != current.get(field):
            reasons.append(f"audit_{field}_mismatch")
    return reasons


def compare_runs(
    golden: dict,
    current: dict,
    max_prob_diff: float,
    max_top1_mismatch: float,
    worst_n: int,
) -> Tuple[Dict[str, Any], int]:
    golden_map = {item["image_id"]: item for item in golden.get("images", [])}
    current_map = {item["image_id"]: item for item in current.get("images", [])}

    diffs: List[Dict[str, Any]] = []
    missing: List[str] = []
    mismatch_count = 0

    for image_id, g_item in golden_map.items():
        c_item = current_map.get(image_id)
        if c_item is None:
            missing.append(image_id)
            continue

        g_probs = np.asarray(g_item["probs"], dtype=np.float32)
        c_probs = np.asarray(c_item["probs"], dtype=np.float32)
        max_diff = float(np.max(np.abs(g_probs - c_probs)))
        top1_mismatch = g_item["top1_label"] != c_item["top1_label"]
        if top1_mismatch:
            mismatch_count += 1

        top1_prob_diff = abs(float(g_item["top1_prob"]) - float(c_item["top1_prob"]))
        g_top3 = {x["label"] for x in g_item["top3"]}
        c_top3 = {x["label"] for x in c_item["top3"]}
        overlap = len(g_top3.intersection(c_top3))

        reasons: List[str] = []
        if g_item.get("input_tensor_hash") != c_item.get("input_tensor_hash"):
            reasons.append("input_tensor_hash_mismatch")
        reasons.extend(
            compare_audit(
                g_item.get("audit", {}),
                c_item.get("audit", {}),
            )
        )
        if not reasons and max_diff > max_prob_diff:
            reasons.append("prob_diff_high_same_input")

        diffs.append(
            {
                "image_id": image_id,
                "filepath": c_item.get("filepath"),
                "max_prob_diff": max_diff,
                "top1_mismatch": top1_mismatch,
                "top1_prob_diff": top1_prob_diff,
                "top3_overlap": overlap,
                "reasons": reasons,
            }
        )

    total = max(1, len(golden_map))
    mismatch_rate = mismatch_count / total

    failures = 0
    if mismatch_rate > max_top1_mismatch:
        failures += 1
    if any(diff["max_prob_diff"] > max_prob_diff for diff in diffs):
        failures += 1

    diffs_sorted = sorted(diffs, key=lambda d: d["max_prob_diff"], reverse=True)
    worst = diffs_sorted[: max(1, worst_n)] if diffs_sorted else []

    report = {
        "total": total,
        "missing": missing,
        "mismatch_rate": mismatch_rate,
        "max_prob_diff_threshold": max_prob_diff,
        "max_top1_mismatch": max_top1_mismatch,
        "max_prob_diff": max((d["max_prob_diff"] for d in diffs), default=0.0),
        "worst": worst,
    }

    return report, failures


def parse_android_probs(path: Path) -> np.ndarray:
    data = load_json(path)
    if isinstance(data, list):
        return np.asarray(data, dtype=np.float32)
    if isinstance(data, dict):
        if "probs" in data:
            return np.asarray(data["probs"], dtype=np.float32)
        if "probabilities" in data:
            return np.asarray(data["probabilities"], dtype=np.float32)
    raise ValueError("Android probs JSON must be a list or contain 'probs'")
