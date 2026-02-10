"""Shared preprocessing for ONNX inference."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except Exception:
    pillow_heif = None

warnings.filterwarnings(
    "ignore",
    message="Palette images with Transparency expressed in bytes should be converted",
    category=UserWarning,
)


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_rgb(img: Image.Image) -> Image.Image:
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


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


def center_crop(img: Image.Image, size: int) -> Image.Image:
    width, height = img.size
    left = max(0, int(round((width - size) / 2.0)))
    top = max(0, int(round((height - size) / 2.0)))
    right = left + size
    bottom = top + size
    return img.crop((left, top, right, bottom))


def resize_square(
    img: Image.Image,
    size: int,
    resample: int = Image.BICUBIC,
) -> Image.Image:
    return img.resize((size, size), resample=resample)


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


def preprocess_image(
    img: Image.Image,
    img_size: int,
    mean: Sequence[float],
    std: Sequence[float],
    mode: str = "center",
    resize_scale: float = 1.15,
) -> np.ndarray:
    img = ensure_rgb(img)
    if mode == "center":
        resized = resize_shorter_side(img, int(round(img_size * resize_scale)))
        cropped = center_crop(resized, img_size)
        return to_tensor(cropped, mean, std)
    if mode == "full":
        resized = resize_square(img, img_size)
        return to_tensor(resized, mean, std)
    raise ValueError(f"Unknown preprocess mode: {mode}")


def preprocess_from_path(
    path: Path,
    img_size: int,
    mean: Sequence[float],
    std: Sequence[float],
    mode: str = "center",
    resize_scale: float = 1.15,
) -> np.ndarray:
    with Image.open(path) as img:
        return preprocess_image(
            img,
            img_size,
            mean,
            std,
            mode=mode,
            resize_scale=resize_scale,
        )


def resolve_labels(label_map: dict) -> list[str]:
    labels = label_map.get("labels")
    if not isinstance(labels, list) or not labels:
        raise ValueError("label_map.json must contain a non-empty 'labels' list")
    return [str(label) for label in labels]


def resolve_mean_std(infer_cfg: dict) -> tuple[list[float], list[float]]:
    mean = infer_cfg.get("mean", [0.485, 0.456, 0.406])
    std = infer_cfg.get("std", [0.229, 0.224, 0.225])
    return [float(x) for x in mean], [float(x) for x in std]


def validate_labels(labels: Iterable[str]) -> None:
    if "other_uncertain" not in labels:
        raise ValueError(
            "Label list must include 'other_uncertain' for rejection handling"
        )
