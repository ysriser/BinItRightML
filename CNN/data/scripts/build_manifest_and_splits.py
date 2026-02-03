"""
Build manifest and train/val/test splits from mixed raw datasets.
Usage:
  python CNN/data/scripts/build_manifest_and_splits.py --config CNN/data/configs/dataset_mix.yaml
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
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


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def has_images(path: Path) -> bool:
    for file in path.rglob("*"):
        if file.is_file() and file.suffix.lower() in IMAGE_EXTS:
            return True
    return False


def candidate_class_dirs(root: Path) -> List[Path]:
    return [d for d in root.iterdir() if d.is_dir() and has_images(d)]


def resolve_class_root(source_cfg: dict) -> Path:
    base = Path(source_cfg.get("path", ""))
    if not base.exists():
        raise FileNotFoundError(f"Source path not found: {base}")
    class_root = source_cfg.get("class_root")
    if class_root:
        root = base / class_root
        if not root.exists():
            raise FileNotFoundError(f"class_root not found: {root}")
        return root

    if candidate_class_dirs(base):
        return base

    subdirs = [d for d in base.iterdir() if d.is_dir() and not d.name.startswith(".")]
    candidates = [d for d in subdirs if candidate_class_dirs(d)]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        f"Cannot auto-detect class_root under {base}. Set class_root in config."
    )


def list_images(class_dir: Path) -> List[Path]:
    return [
        p
        for p in class_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]


def allocate_by_weights(total: int, weights: List[float]) -> List[int]:
    if total <= 0:
        return [0] * len(weights)
    weight_sum = sum(weights)
    if weight_sum <= 0:
        raise ValueError("Mixing weights must sum to > 0.")
    normalized = [w / weight_sum for w in weights]
    raw = [total * w for w in normalized]
    base = [int(math.floor(x)) for x in raw]
    remainder = total - sum(base)
    frac = [x - b for x, b in zip(raw, base)]
    order = sorted(range(len(frac)), key=lambda i: frac[i], reverse=True)
    for i in range(remainder):
        base[order[i % len(base)]] += 1
    return base


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build manifest and splits.")
    p.add_argument(
        "--config",
        type=Path,
        default=Path("CNN/data/configs/dataset_mix.yaml"),
    )
    p.add_argument("--manifest", type=Path, default=Path("CNN/data/tier1_manifest.csv"))
    p.add_argument("--splits-dir", type=Path, default=Path("CNN/data/tier1_splits"))
    p.add_argument("--stats", type=Path, default=Path("CNN/data/tier1_stats.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    root = Path.cwd().resolve()

    final_labels = cfg.get("final_labels") or list(cfg.get("label_mix", {}).keys())
    if not final_labels:
        raise ValueError("final_labels is empty. Check dataset_mix.yaml.")

    sources = cfg.get("sources", {})
    if not sources:
        raise ValueError("No sources found in config.")

    label_mix = cfg.get("label_mix", {})
    caps = cfg.get("caps", {})
    seed = int(cfg.get("seed", 42))

    warnings: List[str] = []
    manifest_rows: List[Dict[str, str]] = []
    pools: Dict[str, Dict[str, List[Dict[str, str]]]] = {
        label: {} for label in final_labels
    }
    assigned: Dict[str, str] = {}

    class_roots: Dict[str, Path] = {}
    for name, meta in sources.items():
        class_roots[name] = resolve_class_root(meta)

    for label in final_labels:
        mixes = label_mix.get(label, [])
        if not mixes:
            raise ValueError(f"Missing label_mix for label '{label}'.")
        for mix in mixes:
            source_name = mix.get("source")
            orig_classes = mix.get("orig_classes", [])
            if source_name not in sources:
                raise ValueError(f"Unknown source '{source_name}' for label '{label}'.")
            if not orig_classes:
                raise ValueError(f"orig_classes is empty for label '{label}' and source '{source_name}'.")

            source_root = class_roots[source_name]
            pools[label].setdefault(source_name, [])
            for orig in orig_classes:
                class_dir = source_root / orig
                if not class_dir.exists():
                    warnings.append(f"Missing class dir: {class_dir}")
                    continue
                for img_path in list_images(class_dir):
                    rel_path = None
                    try:
                        rel_path = img_path.resolve().relative_to(root).as_posix()
                    except ValueError:
                        rel_path = str(img_path.resolve())
                    if rel_path in assigned and assigned[rel_path] != label:
                        warnings.append(f"File used in multiple labels: {rel_path}")
                        continue
                    assigned[rel_path] = label
                    sample = {
                        "filepath": rel_path,
                        "source": source_name,
                        "orig_class": orig,
                        "final_label": label,
                    }
                    pools[label][source_name].append(sample)
                    manifest_rows.append(sample)

    # Apply optional caps per source per label (deterministic).
    for label, sources_map in pools.items():
        for source_name, items in sources_map.items():
            cap = caps.get(source_name, {}).get(label)
            if cap is None:
                continue
            rng = random.Random(seed + stable_int(f"{label}-{source_name}-cap"))
            rng.shuffle(items)
            if len(items) > cap:
                warnings.append(f"Cap applied: {label}/{source_name} {len(items)} -> {cap}")
                pools[label][source_name] = items[: int(cap)]

    # Scale target counts if max_total_images is set.
    target_counts = cfg.get("target_counts", {})
    max_total = cfg.get("max_total_images")
    if max_total:
        max_total = int(max_total)
        total_target = sum(
            int(target_counts[split_name][label_name])
            for split_name in target_counts
            for label_name in target_counts[split_name]
        )
        if total_target > max_total:
            ratio = max_total / total_target
            warnings.append(f"Scaling target counts by ratio {ratio:.4f} to respect max_total_images.")
            for split in target_counts:
                for label in target_counts[split]:
                    target_counts[split][label] = int(math.floor(target_counts[split][label] * ratio))

    # Shuffle pools deterministically.
    for label, sources_map in pools.items():
        for source_name, items in sources_map.items():
            rng = random.Random(seed + stable_int(f"{label}-{source_name}"))
            rng.shuffle(items)

    # Prepare index pointers for each pool.
    indices: Dict[Tuple[str, str], int] = {}
    for label, sources_map in pools.items():
        for source_name in sources_map:
            indices[(label, source_name)] = 0

    def take(label: str, source_name: str, n: int) -> List[Dict[str, str]]:
        pool = pools[label][source_name]
        idx = indices[(label, source_name)]
        picked = pool[idx : idx + n]
        indices[(label, source_name)] = idx + len(picked)
        return picked

    splits = ["train", "val", "test"]
    split_rows: Dict[str, List[Dict[str, str]]] = {s: [] for s in splits}
    split_stats = {s: {"labels": {}, "sources": {}} for s in splits}
    shortages: Dict[str, Dict[str, int]] = {s: {} for s in splits}

    for split in splits:
        if split not in target_counts:
            raise ValueError(f"target_counts missing split '{split}'.")
        for label in final_labels:
            target = int(target_counts[split].get(label, 0))
            mixes = label_mix[label]
            weights = [float(m.get("weight", 0.0)) for m in mixes]
            quotas = allocate_by_weights(target, weights)

            selected: List[Dict[str, str]] = []
            shortage = 0
            for mix, quota in zip(mixes, quotas):
                src = mix["source"]
                got = take(label, src, quota)
                selected.extend(got)
                if len(got) < quota:
                    shortage += quota - len(got)

            if shortage > 0:
                ordered_sources = [m["source"] for m in sorted(mixes, key=lambda m: m["weight"], reverse=True)]
                for src in ordered_sources:
                    if shortage <= 0:
                        break
                    extra = take(label, src, shortage)
                    selected.extend(extra)
                    shortage -= len(extra)

            if shortage > 0:
                warnings.append(f"Underfilled {split}/{label} by {shortage} samples.")
                shortages[split][label] = shortage

            split_rows[split].extend(selected)
            split_stats[split]["labels"][label] = len(selected)
            for item in selected:
                src = item["source"]
                split_stats[split]["sources"][src] = split_stats[split]["sources"].get(src, 0) + 1

        rng = random.Random(seed + stable_int(f"{split}-shuffle"))
        rng.shuffle(split_rows[split])

    # Write manifest.
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "source", "orig_class", "final_label"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    # Write splits.
    args.splits_dir.mkdir(parents=True, exist_ok=True)
    for split in splits:
        out_path = args.splits_dir / f"{split}.csv"
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filepath", "source", "orig_class", "final_label"])
            writer.writeheader()
            writer.writerows(split_rows[split])

    # Optional merged folder with links.
    if cfg.get("make_links", False):
        links_cfg = cfg.get("links", {})
        method = links_cfg.get("method", "auto")
        out_dir = Path(links_cfg.get("out_dir", "CNN/data/merged_tier1"))
        for split in splits:
            for item in split_rows[split]:
                label = item["final_label"]
                src_path = root / item["filepath"]
                dst_dir = out_dir / split / label
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst_path = dst_dir / Path(item["filepath"]).name
                if dst_path.exists():
                    continue
                created = False
                if method in ("auto", "symlink"):
                    try:
                        os.symlink(src_path, dst_path)
                        created = True
                    except OSError:
                        created = False
                if not created and method in ("auto", "hardlink"):
                    try:
                        os.link(src_path, dst_path)
                        created = True
                    except OSError:
                        created = False
                if not created and method in ("auto", "copy"):
                    shutil.copy2(src_path, dst_path)

    stats = {
        "final_labels": final_labels,
        "target_counts": target_counts,
        "actual_counts": split_stats,
        "shortages": shortages,
        "warnings": warnings,
        "class_roots": {k: str(v) for k, v in class_roots.items()},
    }

    args.stats.parent.mkdir(parents=True, exist_ok=True)
    with args.stats.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"Manifest: {args.manifest}")
    print(f"Splits: {args.splits_dir}")
    print(f"Stats: {args.stats}")
    if warnings:
        print(f"Warnings: {len(warnings)} (see tier1_stats.json)")


if __name__ == "__main__":
    main()
