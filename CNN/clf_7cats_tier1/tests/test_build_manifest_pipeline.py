from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from CNN.data.scripts import build_manifest_and_splits as builder


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-image")


def _run_builder(tmp_path: Path, monkeypatch, cfg: dict, run_name: str = "run_test") -> Path:
    cfg_path = tmp_path / "dataset_mix.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    splits_dir = tmp_path / "CNN" / "data" / "tier1_splits"
    args = SimpleNamespace(
        config=cfg_path,
        splits_dir=splits_dir,
        run_name=run_name,
        manifest=None,
        stats=None,
    )

    monkeypatch.setattr(builder, "parse_args", lambda: args)
    monkeypatch.chdir(tmp_path)
    builder.main()

    return splits_dir / run_name


def test_build_manifest_main_creates_expected_files(tmp_path: Path, monkeypatch) -> None:
    _write_image(tmp_path / "raw" / "paper" / "p1.jpg")
    _write_image(tmp_path / "raw" / "trash" / "t1.jpg")

    cfg = {
        "seed": 123,
        "final_labels": ["paper", "other_uncertain"],
        "sources": {
            "rawset": {
                "path": str(tmp_path / "raw"),
            }
        },
        "label_mix": {
            "paper": [
                {"source": "rawset", "orig_classes": ["paper"], "weight": 1.0}
            ],
            "other_uncertain": [
                {"source": "rawset", "orig_classes": ["trash"], "weight": 1.0}
            ],
        },
        "target_counts": {
            "train": {"paper": 1, "other_uncertain": 1},
            "val": {"paper": 0, "other_uncertain": 0},
            "test": {"paper": 0, "other_uncertain": 0},
        },
    }

    run_dir = _run_builder(tmp_path, monkeypatch, cfg)

    assert (run_dir / "manifest.csv").exists()
    assert (run_dir / "train.csv").exists()
    assert (run_dir / "val.csv").exists()
    assert (run_dir / "test.csv").exists()
    assert (run_dir / "stats.json").exists()

    latest_dir = tmp_path / "CNN" / "data" / "tier1_splits" / "latest"
    assert (latest_dir / "manifest.csv").exists()
    assert (latest_dir / "train.csv").exists()

    stats = json.loads((run_dir / "stats.json").read_text(encoding="utf-8"))
    assert stats["actual_counts"]["train"]["labels"]["paper"] == 1
    assert stats["actual_counts"]["train"]["labels"]["other_uncertain"] == 1


def test_build_manifest_main_records_underfill_warning(tmp_path: Path, monkeypatch) -> None:
    _write_image(tmp_path / "raw" / "paper" / "p1.jpg")

    cfg = {
        "seed": 42,
        "final_labels": ["paper"],
        "sources": {"rawset": {"path": str(tmp_path / "raw")}},
        "label_mix": {
            "paper": [{"source": "rawset", "orig_classes": ["paper"], "weight": 1.0}],
        },
        "target_counts": {
            "train": {"paper": 3},
            "val": {"paper": 0},
            "test": {"paper": 0},
        },
    }

    run_dir = _run_builder(tmp_path, monkeypatch, cfg, run_name="run_underfill")
    stats = json.loads((run_dir / "stats.json").read_text(encoding="utf-8"))

    assert stats["shortages"]["train"]["paper"] > 0
    assert any("Underfilled train/paper" in w for w in stats["warnings"])


def test_build_manifest_main_make_links_copy_method(tmp_path: Path, monkeypatch) -> None:
    _write_image(tmp_path / "raw" / "paper" / "p1.jpg")

    cfg = {
        "seed": 9,
        "final_labels": ["paper"],
        "sources": {"rawset": {"path": str(tmp_path / "raw")}},
        "label_mix": {
            "paper": [{"source": "rawset", "orig_classes": ["paper"], "weight": 1.0}],
        },
        "target_counts": {
            "train": {"paper": 1},
            "val": {"paper": 0},
            "test": {"paper": 0},
        },
        "make_links": True,
        "links": {
            "method": "copy",
            "out_dir": str(tmp_path / "merged"),
        },
    }

    _run_builder(tmp_path, monkeypatch, cfg, run_name="run_links")

    copied = list((tmp_path / "merged" / "train" / "paper").glob("*.jpg"))
    assert len(copied) == 1
