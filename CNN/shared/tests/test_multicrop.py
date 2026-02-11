from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from CNN.shared import multicrop


def test_multicrop_triggers_extra_crops() -> None:
    img = Image.new("RGB", (256, 256), color=(128, 128, 128))

    def infer_fn(_: np.ndarray, crop_name: str | None) -> np.ndarray:
        if crop_name == "center":
            return np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        if crop_name == "zoom_center":
            return np.array([[5.0, 0.0, 0.0]], dtype=np.float32)
        return np.array([[0.0, 5.0, 0.0]], dtype=np.float32)

    cfg = {
        "enabled": True,
        "trigger": {"conf": 0.9, "margin": 0.1},
        "extra_crops": ["zoom_center", "full_resize"],
        "combine": "best_score",
    }

    probs, meta = multicrop.run_multicrop(
        img=img,
        infer_fn=infer_fn,
        img_size=224,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        cfg=cfg,
    )

    assert meta["multicrop_used"] is True
    assert meta["selected_crop"] in {"zoom_center", "full_resize"}
    assert int(np.argmax(probs)) in {0, 1}


def test_center_zoom_rejects_invalid_ratio() -> None:
    img = Image.new("RGB", (64, 64), color=(10, 10, 10))
    with pytest.raises(ValueError, match="zoom_ratio"):
        multicrop._center_zoom(img, 1.5)


def test_multicrop_avg_mode_and_fallback_infer_signature() -> None:
    img = Image.new("RGB", (192, 160), color=(100, 100, 100))

    def infer_fn_typeerror_then_none(tensor: np.ndarray, crop_name: str | None) -> np.ndarray:
        if crop_name is not None:
            raise TypeError("legacy signature path")
        if tensor.shape[2] == tensor.shape[3]:
            return np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        return np.array([[0.0, 1.0, 0.0]], dtype=np.float32)

    cfg = {
        "enabled": True,
        "trigger": {"conf": 1.1},
        "extra_crops": ["full_resize"],
        "combine": "avg",
    }

    probs, meta = multicrop.run_multicrop(
        img=img,
        infer_fn=infer_fn_typeerror_then_none,
        img_size=128,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        cfg=cfg,
    )

    assert meta["multicrop_used"] is True
    assert meta["selected_crop"] == "avg"
    assert probs.shape == (3,)


def test_make_crop_and_combine_mode_invalid_values() -> None:
    img = Image.new("RGB", (120, 120), color=(40, 40, 40))

    with pytest.raises(ValueError, match="Unknown crop"):
        multicrop._make_crop(img, "unknown", 64, 1.0, 0.9)

    def infer_fn(_: np.ndarray, _name: str | None) -> np.ndarray:
        return np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

    cfg = {
        "enabled": True,
        "trigger": {"conf": 1.1},
        "extra_crops": ["full_resize"],
        "combine": "invalid_mode",
    }

    with pytest.raises(ValueError, match="Unknown combine mode"):
        multicrop.run_multicrop(
            img=img,
            infer_fn=infer_fn,
            img_size=64,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            cfg=cfg,
        )