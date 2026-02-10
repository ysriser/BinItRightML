from __future__ import annotations

import argparse
import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image


class TinyModel(torch.nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.pool = torch.nn.AdaptiveAvgPool2d((16, 16))
        self.flatten = torch.nn.Flatten()
        self.classifier = torch.nn.Linear(3 * 16 * 16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.flatten(x)
        return self.classifier(x)


def _import_export_onnx(monkeypatch):
    if "CNN.clf_7cats_tier1.export_onnx" in sys.modules:
        del sys.modules["CNN.clf_7cats_tier1.export_onnx"]

    fake_timm = types.ModuleType("timm")

    def create_model(_backbone: str, pretrained: bool, num_classes: int):
        _ = pretrained
        return TinyModel(num_classes)

    fake_timm.create_model = create_model
    monkeypatch.setitem(sys.modules, "timm", fake_timm)

    module = importlib.import_module("CNN.clf_7cats_tier1.export_onnx")
    return importlib.reload(module)


def test_load_json_softmax_and_topk(tmp_path: Path, monkeypatch) -> None:
    mod = _import_export_onnx(monkeypatch)

    cfg_path = tmp_path / "m.json"
    cfg_path.write_text(json.dumps({"x": 1}), encoding="utf-8")
    assert mod.load_json(cfg_path)["x"] == 1

    try:
        mod.load_json(tmp_path / "missing.json")
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass

    probs = mod.softmax_np(np.array([[2.0, 1.0, 0.0]], dtype=np.float32))[0]
    assert np.isclose(probs.sum(), 1.0)
    top = mod.topk(probs, ["a", "b", "c"], 2)
    assert len(top) == 2
    assert top[0][0] == "a"


def test_export_onnx_passes_dynamic_axes(monkeypatch, tmp_path: Path) -> None:
    mod = _import_export_onnx(monkeypatch)
    captured = {}

    def fake_export(
        model,
        dummy,
        out_path,
        input_names,
        output_names,
        opset_version,
        do_constant_folding,
        dynamic_axes,
    ):
        captured["input_shape"] = tuple(dummy.shape)
        captured["out_path"] = out_path
        captured["dynamic_axes"] = dynamic_axes

    monkeypatch.setattr(torch.onnx, "export", fake_export)

    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 16 * 16, 2))
    out_path = tmp_path / "m.onnx"
    mod.export_onnx(model, img_size=16, out_path=out_path, opset=17, dynamic_batch=True)

    assert captured["input_shape"] == (1, 3, 16, 16)
    assert captured["dynamic_axes"]["input"][0] == "batch"
    assert captured["dynamic_axes"]["logits"][0] == "batch"


def _prepare_model_artifacts(tmp_path: Path) -> Path:
    labels = ["paper", "plastic"]
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "label_map.json").write_text(json.dumps({"labels": labels}), encoding="utf-8")
    (model_dir / "infer_config.json").write_text(
        json.dumps(
            {
                "backbone": "efficientnet_b0",
                "img_size": 32,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            }
        ),
        encoding="utf-8",
    )
    torch.save(TinyModel(num_classes=len(labels)).state_dict(), model_dir / "tier1_best.pt")
    return model_dir


def test_main_without_check_image(monkeypatch, tmp_path: Path) -> None:
    mod = _import_export_onnx(monkeypatch)
    model_dir = _prepare_model_artifacts(tmp_path)

    def fake_export(_model, _img_size, out_path: Path, _opset, _dynamic_batch):
        out_path.write_bytes(b"onnx")

    monkeypatch.setattr(mod, "export_onnx", fake_export)
    monkeypatch.setattr(
        mod,
        "parse_args",
        lambda: argparse.Namespace(
            model_dir=model_dir,
            output=None,
            opset=17,
            dynamic_batch=False,
            check_image=None,
            warmup=1,
            runs=1,
        ),
    )

    mod.main()

    assert (model_dir / "tier1.onnx").exists()


def test_main_with_check_image_branch(monkeypatch, tmp_path: Path) -> None:
    mod = _import_export_onnx(monkeypatch)
    model_dir = _prepare_model_artifacts(tmp_path)
    check_img = tmp_path / "check.jpg"
    Image.new("RGB", (40, 40), color=(120, 130, 140)).save(check_img)

    def fake_export(_model, _img_size, out_path: Path, _opset, _dynamic_batch):
        out_path.write_bytes(b"onnx")

    class _Input:
        name = "input"

    class _Sess:
        def __init__(self, _path: str, providers):
            self.providers = providers

        def get_inputs(self):
            return [_Input()]

        def run(self, _outputs, _inputs):
            return [np.array([[0.2, 1.0]], dtype=np.float32)]

    fake_ort = types.ModuleType("onnxruntime")
    fake_ort.InferenceSession = _Sess

    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setattr(mod, "export_onnx", fake_export)
    monkeypatch.setattr(
        mod,
        "parse_args",
        lambda: argparse.Namespace(
            model_dir=model_dir,
            output=model_dir / "exported.onnx",
            opset=17,
            dynamic_batch=True,
            check_image=check_img,
            warmup=1,
            runs=1,
        ),
    )

    mod.main()

    assert (model_dir / "exported.onnx").exists()
