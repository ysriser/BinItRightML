"""
Build manifest and train/val/test splits from mixed raw datasets.
Usage:
  python CNN/data/scripts/build_manifest_and_splits.py --config CNN/data/configs/dataset_mix.yaml
  python CNN/data/scripts/build_manifest_and_splits.py --run-name 20260203_a
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
from datetime import datetime, timezone
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
    p.add_argument("--splits-dir", type=Path, default=Path("CNN/data/tier1_splits"))
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--stats", type=Path, default=None)
    return p.parse_args()


def init_collection_state(
    final_labels: List[str],
) -> Tuple[
    List[str],
    List[Dict[str, str]],
    Dict[str, Dict[str, List[Dict[str, str]]]],
    Dict[str, Dict[str, List[Dict[str, str]]]],
    Dict[str, str],
]:
    warnings: List[str] = []
    manifest_rows: List[Dict[str, str]] = []
    pools: Dict[str, Dict[str, List[Dict[str, str]]]] = {label: {} for label in final_labels}
    fallback_pools: Dict[str, Dict[str, List[Dict[str, str]]]] = {label: {} for label in final_labels}
    assigned: Dict[str, str] = {}
    return warnings, manifest_rows, pools, fallback_pools, assigned


def resolve_all_class_roots(sources: dict) -> Dict[str, Dict[str, List[Path]]]:
    class_roots: Dict[str, Dict[str, List[Path]]] = {}
    for name, meta in sources.items():
        primary_root, fallback_roots = resolve_class_roots(meta)
        class_roots[name] = {
            "primary": [primary_root],
            "fallback": fallback_roots,
        }
    return class_roots


def _resolve_relative_path(root: Path, img_path: Path) -> str:
    try:
        return img_path.resolve().relative_to(root).as_posix()
    except ValueError:
        return str(img_path.resolve())


def _append_samples_from_class_dir(
    class_dir: Path,
    source_name: str,
    orig: str,
    label: str,
    root: Path,
    assigned: Dict[str, str],
    warnings: List[str],
    out_rows: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    if not class_dir.exists():
        warnings.append(f"Missing class dir: {class_dir}")
        return items

    for img_path in list_images(class_dir):
        rel_path = _resolve_relative_path(root, img_path)
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
        items.append(sample)
        out_rows.append(sample)
    return items


def _collect_mix_from_roots(
    source_name: str,
    orig_classes: List[str],
    label: str,
    root_paths: List[Path],
    repo_root: Path,
    assigned: Dict[str, str],
    warnings: List[str],
    manifest_rows: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    collected: List[Dict[str, str]] = []
    for root_path in root_paths:
        for orig in orig_classes:
            class_dir = root_path / orig
            collected.extend(
                _append_samples_from_class_dir(
                    class_dir=class_dir,
                    source_name=source_name,
                    orig=orig,
                    label=label,
                    root=repo_root,
                    assigned=assigned,
                    warnings=warnings,
                    out_rows=manifest_rows,
                )
            )
    return collected


def collect_samples_by_label(
    final_labels: List[str],
    sources: dict,
    label_mix: dict,
    class_roots: Dict[str, Dict[str, List[Path]]],
    root: Path,
    warnings: List[str],
    manifest_rows: List[Dict[str, str]],
    pools: Dict[str, Dict[str, List[Dict[str, str]]]],
    fallback_pools: Dict[str, Dict[str, List[Dict[str, str]]]],
    assigned: Dict[str, str],
) -> None:
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

            source_root = class_roots[source_name]["primary"][0]
            pools[label].setdefault(source_name, [])
            pools[label][source_name].extend(
                _collect_mix_from_roots(
                    source_name=source_name,
                    orig_classes=orig_classes,
                    label=label,
                    root_paths=[source_root],
                    repo_root=root,
                    assigned=assigned,
                    warnings=warnings,
                    manifest_rows=manifest_rows,
                )
            )

            fallback_roots = class_roots[source_name]["fallback"]
            if not fallback_roots:
                continue
            fallback_pools[label].setdefault(source_name, [])
            fallback_pools[label][source_name].extend(
                _collect_mix_from_roots(
                    source_name=source_name,
                    orig_classes=orig_classes,
                    label=label,
                    root_paths=fallback_roots,
                    repo_root=root,
                    assigned=assigned,
                    warnings=warnings,
                    manifest_rows=manifest_rows,
                )
            )


def apply_caps_to_pools(
    pools: Dict[str, Dict[str, List[Dict[str, str]]]],
    caps: dict,
    seed: int,
    warnings: List[str],
) -> None:
    for label, sources_map in pools.items():
        for source_name, items in sources_map.items():
            cap = caps.get(source_name, {}).get(label)
            if cap is None:
                continue
            rng = random.Random(seed + stable_int(f"{label}-{source_name}-cap"))
            rng.shuffle(items)
            if len(items) <= cap:
                continue
            warnings.append(f"Cap applied: {label}/{source_name} {len(items)} -> {cap}")
            pools[label][source_name] = items[: int(cap)]


def scale_target_counts_if_needed(target_counts: dict, max_total: int | None, warnings: List[str]) -> None:
    if not max_total:
        return
    total_target = sum(
        int(target_counts[split_name][label_name])
        for split_name in target_counts
        for label_name in target_counts[split_name]
    )
    if total_target <= max_total:
        return
    ratio = max_total / total_target
    warnings.append(f"Scaling target counts by ratio {ratio:.4f} to respect max_total_images.")
    for split in target_counts:
        for label in target_counts[split]:
            target_counts[split][label] = int(math.floor(target_counts[split][label] * ratio))


def shuffle_all_pools(
    pools: Dict[str, Dict[str, List[Dict[str, str]]]],
    fallback_pools: Dict[str, Dict[str, List[Dict[str, str]]]],
    seed: int,
) -> None:
    for label, sources_map in pools.items():
        for source_name, items in sources_map.items():
            rng = random.Random(seed + stable_int(f"{label}-{source_name}"))
            rng.shuffle(items)
        for source_name, items in fallback_pools.get(label, {}).items():
            rng = random.Random(seed + stable_int(f"{label}-{source_name}-fallback"))
            rng.shuffle(items)


def init_pool_indices(
    pools: Dict[str, Dict[str, List[Dict[str, str]]]],
) -> Tuple[Dict[Tuple[str, str], int], Dict[Tuple[str, str], int]]:
    indices: Dict[Tuple[str, str], int] = {}
    fallback_indices: Dict[Tuple[str, str], int] = {}
    for label, sources_map in pools.items():
        for source_name in sources_map:
            indices[(label, source_name)] = 0
            fallback_indices[(label, source_name)] = 0
    return indices, fallback_indices


def take_from_pools(
    label: str,
    source_name: str,
    n: int,
    pools: Dict[str, Dict[str, List[Dict[str, str]]]],
    fallback_pools: Dict[str, Dict[str, List[Dict[str, str]]]],
    indices: Dict[Tuple[str, str], int],
    fallback_indices: Dict[Tuple[str, str], int],
) -> List[Dict[str, str]]:
    pool = pools[label][source_name]
    idx = indices[(label, source_name)]
    picked = pool[idx : idx + n]
    indices[(label, source_name)] = idx + len(picked)
    remaining = n - len(picked)
    if remaining <= 0:
        return picked

    fb_pool = fallback_pools.get(label, {}).get(source_name, [])
    fb_idx = fallback_indices.get((label, source_name), 0)
    fb_picked = fb_pool[fb_idx : fb_idx + remaining]
    fallback_indices[(label, source_name)] = fb_idx + len(fb_picked)
    picked.extend(fb_picked)
    return picked


def allocate_split_for_label(
    split: str,
    label: str,
    target_counts: dict,
    label_mix: dict,
    pools: Dict[str, Dict[str, List[Dict[str, str]]]],
    fallback_pools: Dict[str, Dict[str, List[Dict[str, str]]]],
    indices: Dict[Tuple[str, str], int],
    fallback_indices: Dict[Tuple[str, str], int],
    warnings: List[str],
) -> Tuple[List[Dict[str, str]], int]:
    target = int(target_counts[split].get(label, 0))
    mixes = label_mix[label]
    weights = [float(m.get("weight", 0.0)) for m in mixes]
    quotas = allocate_by_weights(target, weights)

    selected: List[Dict[str, str]] = []
    shortage = 0
    for mix, quota in zip(mixes, quotas):
        src = mix["source"]
        got = take_from_pools(label, src, quota, pools, fallback_pools, indices, fallback_indices)
        selected.extend(got)
        shortage += max(0, quota - len(got))

    if shortage > 0:
        ordered_sources = [m["source"] for m in sorted(mixes, key=lambda m: m["weight"], reverse=True)]
        for src in ordered_sources:
            if shortage <= 0:
                break
            extra = take_from_pools(label, src, shortage, pools, fallback_pools, indices, fallback_indices)
            selected.extend(extra)
            shortage -= len(extra)

    if shortage > 0:
        warnings.append(f"Underfilled {split}/{label} by {shortage} samples.")
    return selected, max(shortage, 0)


def build_split_rows(
    splits: List[str],
    final_labels: List[str],
    target_counts: dict,
    label_mix: dict,
    pools: Dict[str, Dict[str, List[Dict[str, str]]]],
    fallback_pools: Dict[str, Dict[str, List[Dict[str, str]]]],
    seed: int,
    warnings: List[str],
) -> Tuple[
    Dict[str, List[Dict[str, str]]],
    Dict[str, Dict[str, Dict[str, int]]],
    Dict[str, Dict[str, int]],
]:
    split_rows: Dict[str, List[Dict[str, str]]] = {s: [] for s in splits}
    split_stats = {s: {"labels": {}, "sources": {}} for s in splits}
    shortages: Dict[str, Dict[str, int]] = {s: {} for s in splits}
    indices, fallback_indices = init_pool_indices(pools)

    for split in splits:
        if split not in target_counts:
            raise ValueError(f"target_counts missing split '{split}'.")
        for label in final_labels:
            selected, shortage = allocate_split_for_label(
                split=split,
                label=label,
                target_counts=target_counts,
                label_mix=label_mix,
                pools=pools,
                fallback_pools=fallback_pools,
                indices=indices,
                fallback_indices=fallback_indices,
                warnings=warnings,
            )
            if shortage > 0:
                shortages[split][label] = shortage
            split_rows[split].extend(selected)
            split_stats[split]["labels"][label] = len(selected)
            for item in selected:
                src = item["source"]
                split_stats[split]["sources"][src] = split_stats[split]["sources"].get(src, 0) + 1

        rng = random.Random(seed + stable_int(f"{split}-shuffle"))
        rng.shuffle(split_rows[split])

    return split_rows, split_stats, shortages


def write_manifest_and_splits(
    manifest_path: Path,
    run_dir: Path,
    splits: List[str],
    manifest_rows: List[Dict[str, str]],
    split_rows: Dict[str, List[Dict[str, str]]],
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "source", "orig_class", "final_label"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    for split in splits:
        out_path = run_dir / f"{split}.csv"
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filepath", "source", "orig_class", "final_label"])
            writer.writeheader()
            writer.writerows(split_rows[split])


def maybe_create_links(
    cfg: dict,
    splits: List[str],
    split_rows: Dict[str, List[Dict[str, str]]],
    root: Path,
) -> None:
    if not cfg.get("make_links", False):
        return
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


def build_stats_payload(
    final_labels: List[str],
    target_counts: dict,
    split_stats: dict,
    shortages: dict,
    warnings: List[str],
    class_roots: Dict[str, Dict[str, List[Path]]],
) -> dict:
    return {
        "final_labels": final_labels,
        "target_counts": target_counts,
        "actual_counts": split_stats,
        "shortages": shortages,
        "warnings": warnings,
        "class_roots": {
            k: {"primary": [str(p) for p in v["primary"]], "fallback": [str(p) for p in v["fallback"]]}
            for k, v in class_roots.items()
        },
    }


def write_stats(stats_path: Path, stats: dict) -> None:
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


def copy_latest_files(args: argparse.Namespace, manifest_path: Path, stats_path: Path, run_dir: Path, splits: List[str]) -> Path:
    latest_dir = args.splits_dir / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, latest_dir / "manifest.csv")
    shutil.copy2(stats_path, latest_dir / "stats.json")
    for split in splits:
        shutil.copy2(run_dir / f"{split}.csv", latest_dir / f"{split}.csv")
    return latest_dir


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
    target_counts = cfg.get("target_counts", {})
    max_total_raw = cfg.get("max_total_images")
    max_total = int(max_total_raw) if max_total_raw else None

    warnings, manifest_rows, pools, fallback_pools, assigned = init_collection_state(final_labels)
    class_roots = resolve_all_class_roots(sources)

    collect_samples_by_label(
        final_labels=final_labels,
        sources=sources,
        label_mix=label_mix,
        class_roots=class_roots,
        root=root,
        warnings=warnings,
        manifest_rows=manifest_rows,
        pools=pools,
        fallback_pools=fallback_pools,
        assigned=assigned,
    )
    apply_caps_to_pools(pools, caps, seed, warnings)
    scale_target_counts_if_needed(target_counts, max_total, warnings)
    shuffle_all_pools(pools, fallback_pools, seed)

    splits = ["train", "val", "test"]
    split_rows, split_stats, shortages = build_split_rows(
        splits=splits,
        final_labels=final_labels,
        target_counts=target_counts,
        label_mix=label_mix,
        pools=pools,
        fallback_pools=fallback_pools,
        seed=seed,
        warnings=warnings,
    )

    # Write manifest.
    run_name = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = args.splits_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or (run_dir / "manifest.csv")
    stats_path = args.stats or (run_dir / "stats.json")

    write_manifest_and_splits(
        manifest_path=manifest_path,
        run_dir=run_dir,
        splits=splits,
        manifest_rows=manifest_rows,
        split_rows=split_rows,
    )
    maybe_create_links(cfg=cfg, splits=splits, split_rows=split_rows, root=root)

    stats = build_stats_payload(
        final_labels=final_labels,
        target_counts=target_counts,
        split_stats=split_stats,
        shortages=shortages,
        warnings=warnings,
        class_roots=class_roots,
    )
    write_stats(stats_path, stats)

    latest_dir = copy_latest_files(args, manifest_path, stats_path, run_dir, splits)

    print(f"Run: {run_dir}")
    print(f"Latest: {latest_dir}")
    if warnings:
        print(f"Warnings: {len(warnings)} (see stats.json in run folder)")


if __name__ == "__main__":
    main()
