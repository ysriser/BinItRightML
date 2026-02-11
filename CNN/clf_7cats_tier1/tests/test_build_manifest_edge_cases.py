from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from CNN.data.scripts import build_manifest_and_splits as builder


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-image")


def _run_builder(
    tmp_path: Path,
    monkeypatch,
    cfg: dict,
    *,
    manifest: Path | None = None,
    stats: Path | None = None,
    run_name: str = "edge_run",
) -> Path:
    cfg_path = tmp_path / "dataset_mix.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    splits_dir = tmp_path / "CNN" / "data" / "tier1_splits"
    args = SimpleNamespace(
        config=cfg_path,
        splits_dir=splits_dir,
        run_name=run_name,
        manifest=manifest,
        stats=stats,
    )
    monkeypatch.setattr(builder, "parse_args", lambda: args)
    monkeypatch.chdir(tmp_path)
    builder.main()
    return splits_dir / run_name


def test_parse_args_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["build_manifest_and_splits.py"])
    args = builder.parse_args()

    assert args.config == Path("CNN/data/configs/dataset_mix.yaml")
    assert args.splits_dir == Path("CNN/data/tier1_splits")
    assert args.run_name is None


def test_resolve_class_roots_error_branches(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        builder.resolve_class_roots({"path": str(tmp_path / "missing")})

    root = tmp_path / "dataset"
    root.mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileNotFoundError):
        builder.resolve_class_roots({"path": str(root), "class_root": "not-there"})

    (root / "a").mkdir(parents=True, exist_ok=True)
    (root / "b").mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError, match="Cannot auto-detect class_root"):
        builder.resolve_class_roots({"path": str(root)})


def test_main_cap_and_fallback_fill(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = tmp_path / "raw"
    for i in range(5):
        _write_image(base / "primary" / "paper" / f"p{i}.jpg")
    for i in range(3):
        _write_image(base / "fallback1" / "paper" / f"fb{i}.jpg")

    cfg = {
        "seed": 7,
        "final_labels": ["paper"],
        "sources": {
            "rawset": {
                "path": str(base),
                "class_root": "primary",
                "fallback_roots": ["fallback1", "fallback-missing"],
            }
        },
        "label_mix": {
            "paper": [
                {"source": "rawset", "orig_classes": ["paper", "missing_cls"], "weight": 1.0}
            ]
        },
        "caps": {"rawset": {"paper": 2}},
        "target_counts": {
            "train": {"paper": 4},
            "val": {"paper": 0},
            "test": {"paper": 0},
        },
    }

    run_dir = _run_builder(tmp_path, monkeypatch, cfg, run_name="cap_fb")

    train_rows = list(csv.DictReader((run_dir / "train.csv").open("r", encoding="utf-8")))
    assert len(train_rows) == 4

    stats = json.loads((run_dir / "stats.json").read_text(encoding="utf-8"))
    assert any("Cap applied: paper/rawset" in w for w in stats["warnings"])
    assert any("Missing class dir" in w for w in stats["warnings"])


def test_main_scales_targets_and_supports_custom_manifest_stats(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = tmp_path / "raw"
    for i in range(10):
        _write_image(base / "paper" / f"p{i}.jpg")

    cfg = {
        "seed": 11,
        "final_labels": ["paper"],
        "sources": {"rawset": {"path": str(base)}},
        "label_mix": {
            "paper": [{"source": "rawset", "orig_classes": ["paper"], "weight": 1.0}]
        },
        "target_counts": {
            "train": {"paper": 4},
            "val": {"paper": 2},
            "test": {"paper": 2},
        },
        "max_total_images": 4,
    }

    custom_manifest = tmp_path / "custom_outputs" / "manifest.csv"
    custom_stats = tmp_path / "custom_outputs" / "stats.json"

    _run_builder(
        tmp_path,
        monkeypatch,
        cfg,
        manifest=custom_manifest,
        stats=custom_stats,
        run_name="scaled",
    )

    assert custom_manifest.exists()
    assert custom_stats.exists()

    stats = json.loads(custom_stats.read_text(encoding="utf-8"))
    assert stats["target_counts"]["train"]["paper"] == 2
    assert stats["target_counts"]["val"]["paper"] == 1
    assert stats["target_counts"]["test"]["paper"] == 1
    assert any("Scaling target counts by ratio" in w for w in stats["warnings"])
