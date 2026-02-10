"""
Scan raw dataset folders and write per-source class counts.
Usage:
  python CNN/data/scripts/scan_raw_datasets.py --config CNN/data/configs/dataset_mix.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config must be a YAML mapping.")
    return data


def has_images(path: Path) -> bool:
    for file in path.rglob("*"):
        if file.is_file() and file.suffix.lower() in IMAGE_EXTS:
            return True
    return False


def candidate_class_dirs(root: Path) -> List[Path]:
    return [d for d in root.iterdir() if d.is_dir() and has_images(d)]


def resolve_class_roots(source_cfg: dict) -> Tuple[Path, List[Path]]:
    base = Path(source_cfg.get("path", ""))
    if not base.exists():
        raise FileNotFoundError(f"Source path not found: {base}")
    class_root = source_cfg.get("class_root")
    fallback_roots = source_cfg.get("fallback_roots") or []
    if class_root:
        root = base / class_root
        if not root.exists():
            raise FileNotFoundError(f"class_root not found: {root}")
        fallbacks = []
        for fallback in fallback_roots:
            fallback_path = base / str(fallback)
            if fallback_path.exists():
                fallbacks.append(fallback_path)
        return root, fallbacks

    if candidate_class_dirs(base):
        return base, []

    subdirs = [d for d in base.iterdir() if d.is_dir() and not d.name.startswith(".")]
    candidates = [d for d in subdirs if candidate_class_dirs(d)]
    if len(candidates) == 1:
        return candidates[0], []
    raise ValueError(
        f"Cannot auto-detect class_root under {base}. Set class_root in config."
    )


def scan_source(path: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for class_dir in sorted(path.iterdir()):
        if not class_dir.is_dir():
            continue
        if not has_images(class_dir):
            continue
        count = 0
        for file in class_dir.rglob("*"):
            if file.is_file() and file.suffix.lower() in IMAGE_EXTS:
                count += 1
        counts[class_dir.name] = count
    return counts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scan raw datasets.")
    p.add_argument(
        "--config",
        type=Path,
        default=Path("CNN/data/configs/dataset_mix.yaml"),
    )
    p.add_argument("--out", type=Path, default=Path("CNN/data/raw_stats.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    sources = cfg.get("sources", {})
    if not sources:
        raise ValueError("No sources found in config.")

    report = {"sources": {}, "total_images": 0}

    for name, meta in sources.items():
        class_root, fallback_roots = resolve_class_roots(meta)
        counts = scan_source(class_root)
        total = sum(counts.values())
        root_stats = {str(class_root): counts}
        for fallback in fallback_roots:
            fb_counts = scan_source(fallback)
            root_stats[str(fallback)] = fb_counts
            total += sum(fb_counts.values())
        report["sources"][name] = {
            "path": str(Path(meta.get("path", ""))),
            "class_root": str(class_root),
            "fallback_roots": [str(p) for p in fallback_roots],
            "classes": counts,
            "roots": root_stats,
            "total_images": total,
        }
        report["total_images"] += total
        print(f"{name}: {total} images | {len(counts)} classes | root={class_root}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
