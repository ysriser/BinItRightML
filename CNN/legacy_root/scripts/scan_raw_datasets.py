"""
Scan raw dataset folders and write per-source class counts.
Usage:
  python ml/scripts/scan_raw_datasets.py --config ml/configs/dataset_mix.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config must be a YAML mapping.")
    return data


def scan_source(path: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if not path.exists():
        return counts
    for class_dir in sorted(p for p in path.iterdir() if p.is_dir()):
        count = 0
        for file in class_dir.rglob("*"):
            if file.is_file() and file.suffix.lower() in IMAGE_EXTS:
                count += 1
        counts[class_dir.name] = count
    return counts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scan raw datasets.")
    p.add_argument("--config", type=Path, default=Path("ml/configs/dataset_mix.yaml"))
    p.add_argument("--out", type=Path, default=Path("ml/data/raw_stats.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    sources = cfg.get("sources", {})
    if not sources:
        raise ValueError("No sources found in config.")

    report = {"sources": {}, "total_images": 0}

    for name, meta in sources.items():
        path = Path(meta.get("path", ""))
        counts = scan_source(path)
        total = sum(counts.values())
        report["sources"][name] = {"path": str(path), "classes": counts, "total_images": total}
        report["total_images"] += total
        print(f"{name}: {total} images | {len(counts)} classes")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
