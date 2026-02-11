from __future__ import annotations

import argparse
import importlib
import json
import sys
import types
from pathlib import Path

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


def _import_eval_custom(monkeypatch):
    if "CNN.clf_7cats_tier1.eval_custom" in sys.modules:
        del sys.modules["CNN.clf_7cats_tier1.eval_custom"]

    fake_timm = types.ModuleType("timm")

    def create_model(_backbone: str, pretrained: bool, num_classes: int):
        _ = pretrained
        return TinyModel(num_classes)

    fake_timm.create_model = create_model
    monkeypatch.setitem(sys.modules, "timm", fake_timm)

    module = importlib.import_module("CNN.clf_7cats_tier1.eval_custom")
    return importlib.reload(module)


def test_load_json_and_list_images(tmp_path: Path, monkeypatch) -> None:
    mod = _import_eval_custom(monkeypatch)

    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({"a": 1}), encoding="utf-8")
    assert mod.load_json(cfg_path)["a"] == 1

    missing = tmp_path / "missing.json"
    try:
        mod.load_json(missing)
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass

    (tmp_path / "paper").mkdir()
    (tmp_path / "unknown").mkdir()
    img = Image.new("RGB", (16, 16), color=(1, 2, 3))
    img.save(tmp_path / "paper" / "a.jpg")
    img.save(tmp_path / "unknown" / "b.jpg")

    items = mod.list_images(tmp_path, ["paper", "plastic"])
    assert len(items) == 1
    assert items[0][1] == "paper"


def test_build_transform_and_draw_overlay(monkeypatch) -> None:
    mod = _import_eval_custom(monkeypatch)

    tfm = mod.build_transform(32, [0.5, 0.5, 0.5], [0.2, 0.2, 0.2])
    tensor = tfm(Image.new("RGB", (48, 48), color=(100, 100, 100)))
    assert isinstance(tensor, torch.Tensor)
    assert tuple(tensor.shape) == (3, 32, 32)

    img = Image.new("RGB", (120, 80), color=(20, 30, 40))
    out = mod.draw_overlay(img.copy(), ["true: paper", "pred: plastic"])
    assert out.size == img.size
    assert isinstance(out, Image.Image)


def test_eval_custom_main_smoke(monkeypatch, tmp_path: Path) -> None:
    mod = _import_eval_custom(monkeypatch)

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

    model = TinyModel(num_classes=len(labels))
    torch.save(model.state_dict(), model_dir / "tier1_best.pt")

    data_dir = tmp_path / "data"
    (data_dir / "paper").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (40, 40), color=(50, 60, 70)).save(data_dir / "paper" / "sample.jpg")

    output_dir = tmp_path / "out"

    monkeypatch.setattr(
        mod,
        "parse_args",
        lambda: argparse.Namespace(
            data_dir=data_dir,
            model_dir=model_dir,
            output_dir=output_dir,
            topk=2,
            max_images=0,
            save_images=False,
        ),
    )

    mod.main()

    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "predictions.json").exists()
    assert (output_dir / "misclassified.json").exists()
    assert (output_dir / "report.json").exists()
    assert (output_dir / "report.txt").exists()
