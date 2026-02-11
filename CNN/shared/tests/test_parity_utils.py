from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from CNN.experiments.v1_parity_self_test import parity_utils as u


def test_load_and_save_json_roundtrip(tmp_path: Path) -> None:
    out = tmp_path / "x" / "data.json"
    payload = {"a": 1, "b": [1, 2]}
    u.save_json(out, payload)

    loaded = u.load_json(out)

    assert loaded == payload


def test_load_json_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        u.load_json(tmp_path / "missing.json")


def test_load_labels_validates_non_empty(tmp_path: Path) -> None:
    label_map = tmp_path / "labels.json"
    label_map.write_text(json.dumps({"labels": ["paper", "metal"]}), encoding="utf-8")

    labels = u.load_labels(label_map)

    assert labels == ["paper", "metal"]


def test_load_labels_invalid_raises(tmp_path: Path) -> None:
    label_map = tmp_path / "labels.json"
    label_map.write_text(json.dumps({"labels": []}), encoding="utf-8")

    with pytest.raises(ValueError):
        u.load_labels(label_map)


def test_load_infer_config_defaults(tmp_path: Path) -> None:
    infer = tmp_path / "infer.json"
    infer.write_text(json.dumps({"img_size": 128}), encoding="utf-8")

    cfg = u.load_infer_config(infer, resize_scale=1.2)

    assert cfg.img_size == 128
    assert cfg.resize_scale == 1.2
    assert cfg.mean == [0.485, 0.456, 0.406]


def test_list_images_from_folder_filters_by_extension(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.jpg").write_bytes(b"x")
    (tmp_path / "a" / "y.txt").write_text("n", encoding="utf-8")

    items = u.list_images_from_folder(tmp_path)

    assert [p.name for p in items] == ["x.jpg"]


def test_list_items_requires_source() -> None:
    with pytest.raises(ValueError):
        u.list_items(images_dir=None, manifest_path=None)


def test_list_items_from_images_dir(tmp_path: Path) -> None:
    img = tmp_path / "paper" / "a.jpg"
    img.parent.mkdir(parents=True)
    img.write_bytes(b"fake")

    items = u.list_items(images_dir=tmp_path, manifest_path=None)

    assert len(items) == 1
    assert items[0]["label"] == "paper"


def test_list_items_from_manifest(tmp_path: Path) -> None:
    image = tmp_path / "img.jpg"
    image.write_bytes(b"fake")
    manifest = tmp_path / "m.csv"
    with manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "label"])
        writer.writeheader()
        writer.writerow({"filepath": "img.jpg", "label": "paper"})

    items = u.list_items(images_dir=None, manifest_path=manifest)

    assert len(items) == 1
    assert items[0]["path"] == image
    assert items[0]["label"] == "paper"


def test_list_items_manifest_missing_filepath_column(tmp_path: Path) -> None:
    manifest = tmp_path / "m.csv"
    with manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["x"])
        writer.writeheader()
        writer.writerow({"x": "a"})

    with pytest.raises(ValueError, match="filepath"):
        u.list_items(images_dir=None, manifest_path=manifest)


def test_resize_center_crop_and_tensor() -> None:
    img = Image.new("RGBA", (200, 100), (10, 20, 30, 255))

    resized = u.resize_shorter_side(img, 50)
    cropped, crop_box = u.center_crop(resized, 40)
    tensor = u.to_tensor(cropped, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    assert resized.size == (100, 50)
    assert cropped.size == (40, 40)
    assert len(crop_box) == 4
    assert tensor.shape == (1, 3, 40, 40)


def test_preprocess_with_audit_has_required_fields() -> None:
    img = Image.new("RGB", (120, 80), "white")
    cfg = u.InferConfig(
        img_size=64,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        resize_scale=1.1,
        interpolation="BICUBIC",
    )

    tensor, audit = u.preprocess_with_audit(img, cfg)

    assert tensor.shape == (1, 3, 64, 64)
    assert audit["channel_order"] == "RGB"
    assert audit["interpolation"] == "BICUBIC"


def test_softmax_topk_and_hash_array() -> None:
    logits = np.array([0.1, 1.1, -0.2], dtype=np.float32)
    probs = u.softmax(logits)

    assert pytest.approx(float(np.sum(probs)), rel=1e-6) == 1.0
    top = u.topk(probs, ["a", "b", "c"], 2)
    assert len(top) == 2
    assert top[0]["label"] == "b"

    h1 = u.hash_array(probs)
    h2 = u.hash_array(probs)
    assert h1 == h2


def test_build_session_missing_model_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_ort = SimpleNamespace(
        get_available_providers=lambda: ["CPUExecutionProvider"],
        InferenceSession=lambda *args, **kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    with pytest.raises(FileNotFoundError):
        u.build_session(tmp_path / "missing.onnx")


def test_build_session_with_fake_onnxruntime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"x")

    captured = {}

    class FakeSession:
        def __init__(self, path: str, providers: list[str]) -> None:
            captured["path"] = path
            captured["providers"] = providers

    fake_ort = SimpleNamespace(
        get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
        InferenceSession=FakeSession,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    session = u.build_session(model, prefer_cuda=True)

    assert isinstance(session, FakeSession)
    assert captured["providers"] == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_run_onnx_calls_session() -> None:
    class FakeIo:
        def __init__(self, name: str) -> None:
            self.name = name

    class FakeSession:
        def get_inputs(self):
            return [FakeIo("in")]

        def get_outputs(self):
            return [FakeIo("out")]

        def run(self, output_names, feeds):
            assert output_names == ["out"]
            assert "in" in feeds
            return [np.array([[0.2, 0.8]], dtype=np.float32)]

    out = u.run_onnx(FakeSession(), np.array([[1.0]], dtype=np.float32))
    assert out.shape == (1, 2)


def test_record_for_image_and_build_output(tmp_path: Path) -> None:
    img_path = tmp_path / "sample.png"
    Image.new("RGB", (64, 64), "blue").save(img_path)

    class FakeIo:
        def __init__(self, name: str) -> None:
            self.name = name

    class FakeSession:
        def get_inputs(self):
            return [FakeIo("in")]

        def get_outputs(self):
            return [FakeIo("out")]

        def run(self, output_names, feeds):
            return [np.array([[0.2, 0.8]], dtype=np.float32)]

    cfg = u.InferConfig(64, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], 1.0, "BICUBIC")

    rec = u.record_for_image(
        session=FakeSession(),
        cfg=cfg,
        labels=["paper", "metal"],
        path=img_path,
        label="paper",
        image_id="id1",
        topk_n=2,
    )
    assert rec["top1_label"] == "metal"
    assert "audit" in rec

    output = u.build_output([rec], Path("m.onnx"), Path("i.json"), Path("l.json"), cfg)
    assert output["meta"]["total"] == 1


def test_write_csv_creates_file(tmp_path: Path) -> None:
    csv_path = tmp_path / "out.csv"
    rec = {
        "image_id": "id1",
        "filepath": "x.jpg",
        "label": "paper",
        "top1_label": "paper",
        "top1_prob": 0.9,
        "top3": [{"label": "paper", "p": 0.9}],
        "prob_hash": "a",
        "input_tensor_hash": "b",
        "audit": {
            "original_size": [1, 1],
            "resized_size": [1, 1],
            "crop_box": [0, 0, 1, 1],
            "resize_scale": 1.15,
            "interpolation": "BICUBIC",
        },
    }

    u.write_csv(csv_path, [rec])

    assert csv_path.exists()


def test_compare_audit_and_compare_runs() -> None:
    golden = {
        "images": [
            {
                "image_id": "a",
                "filepath": "a.jpg",
                "top1_label": "paper",
                "top1_prob": 0.9,
                "top3": [{"label": "paper", "p": 0.9}],
                "probs": [0.9, 0.1],
                "input_tensor_hash": "x",
                "audit": {
                    "original_size": [10, 10],
                    "resized_size": [10, 10],
                    "crop_box": [0, 0, 10, 10],
                    "channel_order": "RGB",
                    "pixel_scaling": "0-1",
                    "normalize_mean": [0.1, 0.2, 0.3],
                    "normalize_std": [0.1, 0.2, 0.3],
                    "resize_scale": 1.15,
                    "interpolation": "BICUBIC",
                },
            }
        ]
    }
    current = {
        "images": [
            {
                "image_id": "a",
                "filepath": "a.jpg",
                "top1_label": "metal",
                "top1_prob": 0.7,
                "top3": [{"label": "metal", "p": 0.7}],
                "probs": [0.7, 0.3],
                "input_tensor_hash": "y",
                "audit": {
                    "original_size": [11, 10],
                    "resized_size": [10, 10],
                    "crop_box": [0, 0, 10, 10],
                    "channel_order": "RGB",
                    "pixel_scaling": "0-1",
                    "normalize_mean": [0.1, 0.2, 0.3],
                    "normalize_std": [0.1, 0.2, 0.3],
                    "resize_scale": 1.15,
                    "interpolation": "BICUBIC",
                },
            }
        ]
    }

    report, failures = u.compare_runs(
        golden=golden,
        current=current,
        max_prob_diff=0.02,
        max_top1_mismatch=0.0,
        worst_n=3,
    )

    assert report["total"] == 1
    assert report["mismatch_rate"] == 1.0
    assert failures >= 1
    assert report["worst"][0]["reasons"]


def test_parse_android_probs_variants(tmp_path: Path) -> None:
    list_path = tmp_path / "list.json"
    list_path.write_text(json.dumps([0.1, 0.9]), encoding="utf-8")
    arr = u.parse_android_probs(list_path)
    assert arr.shape == (2,)

    dict_path = tmp_path / "dict.json"
    dict_path.write_text(json.dumps({"probs": [0.2, 0.8]}), encoding="utf-8")
    arr2 = u.parse_android_probs(dict_path)
    assert arr2.shape == (2,)

    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps({"x": [1]}), encoding="utf-8")
    with pytest.raises(ValueError):
        u.parse_android_probs(bad_path)

