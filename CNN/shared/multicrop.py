"""Multi-crop inference utilities."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from PIL import Image

from . import decision, preprocess


CropInferFn = Callable[[np.ndarray, str | None], np.ndarray]


def _center_zoom(img: Image.Image, zoom_ratio: float) -> Image.Image:
    if zoom_ratio <= 0 or zoom_ratio > 1:
        raise ValueError("zoom_ratio must be in (0, 1]")
    width, height = img.size
    crop_w = max(1, int(round(width * zoom_ratio)))
    crop_h = max(1, int(round(height * zoom_ratio)))
    left = max(0, int(round((width - crop_w) / 2.0)))
    top = max(0, int(round((height - crop_h) / 2.0)))
    right = left + crop_w
    bottom = top + crop_h
    return img.crop((left, top, right, bottom))


def _make_crop(
    img: Image.Image,
    crop_name: str,
    img_size: int,
    resize_scale: float,
    zoom_ratio: float,
) -> Image.Image:
    if crop_name == "center":
        resized = preprocess.resize_shorter_side(
            img,
            int(round(img_size * resize_scale)),
        )
        return preprocess.center_crop(resized, img_size)
    if crop_name == "zoom_center":
        zoomed = _center_zoom(img, zoom_ratio)
        resized = preprocess.resize_shorter_side(
            zoomed,
            int(round(img_size * resize_scale)),
        )
        return preprocess.center_crop(resized, img_size)
    if crop_name == "full_resize":
        return preprocess.resize_square(img, img_size)
    raise ValueError(f"Unknown crop: {crop_name}")


def _infer(infer_fn: CropInferFn, tensor: np.ndarray, crop_name: str) -> np.ndarray:
    try:
        return infer_fn(tensor, crop_name)
    except TypeError:
        return infer_fn(tensor, None)


def _should_trigger(metrics: Dict[str, float], trigger_cfg: Dict[str, Any]) -> bool:
    conf_th = trigger_cfg.get("conf")
    margin_th = trigger_cfg.get("margin")
    if conf_th is not None and metrics["max_prob"] < float(conf_th):
        return True
    if margin_th is not None and metrics["margin"] < float(margin_th):
        return True
    return False


def run_multicrop(
    img: Image.Image,
    infer_fn: CropInferFn,
    img_size: int,
    mean: List[float],
    std: List[float],
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    enabled = bool(cfg.get("enabled", True))
    resize_scale = float(cfg.get("resize_scale", 1.15))
    zoom_ratio = float(cfg.get("zoom_ratio", 0.85))
    trigger_cfg = cfg.get("trigger", {}) or {}
    extra_crops = cfg.get("extra_crops", []) or []
    combine_mode = str(cfg.get("combine", "avg"))

    base_crop = _make_crop(img, "center", img_size, resize_scale, zoom_ratio)
    base_tensor = preprocess.to_tensor(base_crop, mean, std)
    base_logits = _infer(infer_fn, base_tensor, "center")
    base_probs = decision.softmax(base_logits)
    base_metrics = decision.compute_metrics(base_probs)

    if not enabled or not extra_crops or not _should_trigger(base_metrics, trigger_cfg):
        return base_probs, {
            "multicrop_used": False,
            "selected_crop": "center",
            "per_crop": {"center": base_metrics},
        }

    crop_results: List[Tuple[str, np.ndarray, Dict[str, float]]] = [
        ("center", base_probs, base_metrics)
    ]
    for crop_name in extra_crops:
        crop_img = _make_crop(img, crop_name, img_size, resize_scale, zoom_ratio)
        tensor = preprocess.to_tensor(crop_img, mean, std)
        logits = _infer(infer_fn, tensor, crop_name)
        probs = decision.softmax(logits)
        metrics = decision.compute_metrics(probs)
        crop_results.append((crop_name, probs, metrics))

    if combine_mode == "avg":
        stacked = np.stack([item[1] for item in crop_results], axis=0)
        combined = stacked.mean(axis=0)
        return combined, {
            "multicrop_used": True,
            "selected_crop": "avg",
            "per_crop": {name: metrics for name, _, metrics in crop_results},
        }

    if combine_mode == "best_score":
        best_name = crop_results[0][0]
        best_probs = crop_results[0][1]
        best_score = crop_results[0][2]["max_prob"] + crop_results[0][2]["margin"]
        for name, probs, metrics in crop_results[1:]:
            score = metrics["max_prob"] + metrics["margin"]
            if score > best_score:
                best_score = score
                best_name = name
                best_probs = probs
        return best_probs, {
            "multicrop_used": True,
            "selected_crop": best_name,
            "per_crop": {name: metrics for name, _, metrics in crop_results},
        }

    raise ValueError(f"Unknown combine mode: {combine_mode}")
