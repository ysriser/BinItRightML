from __future__ import annotations

import argparse
import csv
import importlib
import json
import random
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

    def get_classifier(self) -> torch.nn.Module:
        return self.classifier


def _import_train(monkeypatch):
    if "CNN.clf_7cats_tier1.train" in sys.modules:
        del sys.modules["CNN.clf_7cats_tier1.train"]

    fake_timm = types.ModuleType("timm")

    def create_model(_backbone: str, pretrained: bool, num_classes: int):
        _ = pretrained
        return TinyModel(num_classes)

    fake_timm.create_model = create_model
    fake_timm.data = types.SimpleNamespace(
        resolve_data_config=lambda *_args, **_kwargs: {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        }
    )
    monkeypatch.setitem(sys.modules, "timm", fake_timm)

    module = importlib.import_module("CNN.clf_7cats_tier1.train")
    return importlib.reload(module)


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (32, 32), color=(30, 40, 50))
    img.save(path)


def _write_split_csv(path: Path, rows: list[tuple[Path, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "final_label"])
        writer.writeheader()
        for fp, label in rows:
            writer.writerow({"filepath": str(fp), "final_label": label})


def test_set_seed_and_collect_predictions(monkeypatch) -> None:
    mod = _import_train(monkeypatch)

    mod.set_seed(123)
    v1 = (random.random(), float(np.random.rand()), float(torch.rand(1).item()))
    mod.set_seed(123)
    v2 = (random.random(), float(np.random.rand()), float(torch.rand(1).item()))
    assert v1 == v2

    model = TinyModel(num_classes=3)
    images = torch.rand(4, 3, 8, 8)
    labels = torch.tensor([0, 1, 2, 1], dtype=torch.long)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(images, labels),
        batch_size=2,
        shuffle=False,
    )

    metrics, y_true, y_pred = mod.collect_predictions(
        model,
        loader,
        torch.device("cpu"),
        num_classes=3,
        topk=2,
    )

    assert set(metrics.keys()) == {"top1", "top3", "macro_f1"}
    assert len(y_true) == 4
    assert len(y_pred) == 4


def test_main_smoke_with_minimal_config(monkeypatch, tmp_path: Path) -> None:
    mod = _import_train(monkeypatch)

    labels = ["paper", "plastic"]
    train_img = tmp_path / "data" / "paper_train.jpg"
    val_img = tmp_path / "data" / "paper_val.jpg"
    test_img = tmp_path / "data" / "paper_test.jpg"
    for p in [train_img, val_img, test_img]:
        _write_image(p)

    train_csv = tmp_path / "splits" / "train.csv"
    val_csv = tmp_path / "splits" / "val.csv"
    test_csv = tmp_path / "splits" / "test.csv"
    _write_split_csv(train_csv, [(train_img, "paper")])
    _write_split_csv(val_csv, [(val_img, "paper")])
    _write_split_csv(test_csv, [(test_img, "paper")])

    cfg = {
        "seed": 7,
        "labels": labels,
        "device": "cpu",
        "require_cuda": False,
        "paths": {
            "train_csv": str(train_csv),
            "val_csv": str(val_csv),
            "test_csv": str(test_csv),
            "output_dir": str(tmp_path / "outputs"),
            "model_dir": str(tmp_path / "models"),
        },
        "model": {
            "backbone": "efficientnet_b0",
            "pretrained": False,
            "img_size": 32,
        },
        "data": {
            "batch_size": 1,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "verify_images": False,
            "skip_bad_images": True,
            "max_retry": 2,
        },
        "training": {
            "freeze_epochs": 1,
            "finetune_epochs": 0,
            "grad_clip": 1.0,
            "use_amp": False,
            "early_stop_patience": 0,
            "show_progress": False,
        },
        "optimizer": {
            "lr_freeze": 0.001,
            "lr_finetune": 0.0001,
            "weight_decay": 0.0,
            "betas": [0.9, 0.999],
        },
        "scheduler": {"enabled": False},
        "loss": {"label_smoothing": 0.0},
        "infer": {"topk": 2, "conf_threshold": 0.7, "margin_threshold": 0.05},
        "augment": {
            "scale_min": 0.8,
            "color_jitter": [0.1, 0.1, 0.1, 0.05],
            "perspective_distortion": 0.1,
            "perspective_p": 0.0,
            "blur_kernel": 3,
            "blur_sigma": [0.1, 0.2],
            "random_erasing_p": 0.0,
        },
    }
    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    monkeypatch.setattr(mod, "parse_args", lambda: argparse.Namespace(config=cfg_path))
    monkeypatch.setattr(mod, "save_confusion_from_preds", lambda *args, **kwargs: None)

    mod.main()

    model_dir = Path(cfg["paths"]["model_dir"])
    assert (model_dir / "tier1_best.pt").exists()
    assert (model_dir / "label_map.json").exists()
    assert (model_dir / "infer_config.json").exists()

    output_dir = Path(cfg["paths"]["output_dir"])
    runs = list(output_dir.glob("*"))
    assert runs, "Expected at least one run output folder"
