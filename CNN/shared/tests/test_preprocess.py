from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from CNN.shared import preprocess


def test_load_json_reads_file(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"a": 1}), encoding="utf-8")

    data = preprocess.load_json(path)

    assert data == {"a": 1}


def test_load_json_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        preprocess.load_json(tmp_path / "missing.json")


def test_ensure_rgb_converts_non_rgb() -> None:
    gray = Image.new("L", (8, 8), 255)

    converted = preprocess.ensure_rgb(gray)

    assert converted.mode == "RGB"


def test_resize_shorter_side_keeps_aspect_ratio() -> None:
    img = Image.new("RGB", (300, 200), "white")

    resized = preprocess.resize_shorter_side(img, 100)

    assert resized.size == (150, 100)


def test_center_crop_returns_target_size() -> None:
    img = Image.new("RGB", (150, 100), "white")

    cropped = preprocess.center_crop(img, 64)

    assert cropped.size == (64, 64)


def test_to_tensor_handles_grayscale_and_rgba() -> None:
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    gray = Image.new("L", (10, 10), 128)
    rgba = Image.new("RGBA", (10, 10), (10, 20, 30, 255))

    gray_tensor = preprocess.to_tensor(gray, mean, std)
    rgba_tensor = preprocess.to_tensor(rgba, mean, std)

    assert gray_tensor.shape == (1, 3, 10, 10)
    assert rgba_tensor.shape == (1, 3, 10, 10)
    assert gray_tensor.dtype == np.float32


def test_preprocess_image_center_and_full_modes() -> None:
    img = Image.new("RGB", (320, 240), "white")
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    center = preprocess.preprocess_image(img, 224, mean, std, mode="center")
    full = preprocess.preprocess_image(img, 224, mean, std, mode="full")

    assert center.shape == (1, 3, 224, 224)
    assert full.shape == (1, 3, 224, 224)


def test_preprocess_image_unknown_mode_raises() -> None:
    img = Image.new("RGB", (100, 100), "white")
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    with pytest.raises(ValueError, match="Unknown preprocess mode"):
        preprocess.preprocess_image(img, 64, mean, std, mode="bad-mode")


def test_resolve_labels_and_validate_labels() -> None:
    labels = preprocess.resolve_labels({"labels": ["paper", "other_uncertain"]})

    preprocess.validate_labels(labels)

    assert labels == ["paper", "other_uncertain"]


def test_validate_labels_requires_other_uncertain() -> None:
    with pytest.raises(ValueError, match="other_uncertain"):
        preprocess.validate_labels(["paper", "plastic"])


def test_resolve_mean_std_defaults() -> None:
    mean, std = preprocess.resolve_mean_std({})

    assert mean == [0.485, 0.456, 0.406]
    assert std == [0.229, 0.224, 0.225]
