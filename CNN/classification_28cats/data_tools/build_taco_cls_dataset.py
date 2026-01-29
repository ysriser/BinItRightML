import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build TACO super-28 classification crops from COCO.")
    p.add_argument("--images_dir", type=Path, required=True)
    p.add_argument("--coco_json", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, default=Path("ml/data/taco_super28_cls"))
    p.add_argument("--split_ratios", type=float, nargs=3, default=[0.8, 0.1, 0.1])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pad_ratio", type=float, default=0.25)
    p.add_argument("--min_box_area_ratio", type=float, default=0.001)
    p.add_argument("--use_masks", action="store_true")
    p.add_argument("--label_map", type=Path, default=Path("ml/classification_28cats/configs/label_map_taco_super28.yaml"))
    p.add_argument("--suggest_keep", type=float, default=0.8, help="Target keep ratio for suggested threshold.")
    p.add_argument("--hist_path", type=Path, default=None, help="Where to save area ratio histogram PNG.")
    return p.parse_args()


def load_coco(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_label_map(categories: List[dict]) -> Tuple[Dict[str, str], List[str]]:
    fine_to_super: Dict[str, str] = {}
    super_set = set()
    for cat in categories:
        name = cat.get("name")
        super_name = cat.get("supercategory")
        if not name or not super_name:
            continue
        fine_to_super[name] = super_name
        super_set.add(super_name)
    return fine_to_super, sorted(super_set)


def save_label_map(path: Path, fine_to_super: Dict[str, str], super_classes: List[str]) -> None:
    payload = {
        "super_classes": super_classes,
        "fine_to_super": fine_to_super,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Auto-generated from COCO categories\n")
        yaml.safe_dump(payload, f, sort_keys=False)


def split_image_ids(image_ids: List[int], ratios: List[float], seed: int) -> Dict[int, str]:
    if not math.isclose(sum(ratios), 1.0, abs_tol=1e-3):
        raise ValueError("split_ratios must sum to 1.0")
    random.Random(seed).shuffle(image_ids)
    n = len(image_ids)
    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)
    train_ids = set(image_ids[:n_train])
    val_ids = set(image_ids[n_train : n_train + n_val])
    test_ids = set(image_ids[n_train + n_val :])
    split_map = {}
    for img_id in train_ids:
        split_map[img_id] = "train"
    for img_id in val_ids:
        split_map[img_id] = "val"
    for img_id in test_ids:
        split_map[img_id] = "test"
    return split_map


def create_mask_from_segmentation(segmentation, size: Tuple[int, int]):
    if isinstance(segmentation, list):
        mask = Image.new("L", size, 0)
        draw = ImageDraw.Draw(mask)
        for poly in segmentation:
            if len(poly) >= 6:
                xy = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
                draw.polygon(xy, outline=255, fill=255)
        return mask
    return None


def quantile(sorted_values: List[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    q = min(max(q, 0.0), 1.0)
    idx = int(round(q * (len(sorted_values) - 1)))
    return float(sorted_values[idx])


def main() -> None:
    args = parse_args()
    coco = load_coco(args.coco_json)

    categories = coco.get("categories", [])
    categories_by_id = {c["id"]: c for c in categories}
    fine_to_super, super_classes = build_label_map(categories)
    save_label_map(args.label_map, fine_to_super, super_classes)

    images = {img["id"]: img for img in coco.get("images", [])}
    anns_by_image: Dict[int, List[dict]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        anns_by_image[ann["image_id"]].append(ann)

    split_map = split_image_ids(list(images.keys()), args.split_ratios, args.seed)

    out_dir = args.out_dir
    area_ratios: List[float] = []
    # Create empty class folders for all splits to keep class_to_idx consistent.
    for split in ("train", "val", "test"):
        for super_name in super_classes:
            (out_dir / split / super_name).mkdir(parents=True, exist_ok=True)
    stats = {
        "total_annotations": 0,
        "kept": 0,
        "filtered_small": 0,
        "filtered_invalid": 0,
        "missing_image": 0,
        "unknown_category": 0,
        "per_super": defaultdict(int),
        "super_classes": super_classes,
    }

    for img_id, ann_list in anns_by_image.items():
        img_info = images.get(img_id)
        if img_info is None:
            stats["missing_image"] += len(ann_list)
            continue

        split = split_map.get(img_id)
        if split is None:
            continue

        img_path = args.images_dir / img_info["file_name"]
        if not img_path.exists():
            stats["missing_image"] += len(ann_list)
            continue

        with Image.open(img_path) as im:
            im = im.convert("RGB")
            img_w, img_h = im.size

            for ann in ann_list:
                stats["total_annotations"] += 1
                bbox = ann.get("bbox", [])
                if len(bbox) != 4:
                    stats["filtered_invalid"] += 1
                    continue
                x, y, w, h = bbox
                if w <= 1 or h <= 1:
                    stats["filtered_invalid"] += 1
                    continue

                area_ratio = (w * h) / (img_w * img_h)
                area_ratios.append(area_ratio)
                if area_ratio < args.min_box_area_ratio:
                    stats["filtered_small"] += 1
                    continue

                cat_id = ann.get("category_id")
                cat = categories_by_id.get(cat_id)
                if not cat:
                    stats["unknown_category"] += 1
                    continue

                fine_name = cat["name"]
                super_name = fine_to_super.get(fine_name)
                if super_name is None:
                    raise ValueError(f"Unknown fine category: {fine_name}")

                pad_w = w * args.pad_ratio
                pad_h = h * args.pad_ratio
                x0 = max(0, int(x - pad_w))
                y0 = max(0, int(y - pad_h))
                x1 = min(img_w, int(x + w + pad_w))
                y1 = min(img_h, int(y + h + pad_h))

                # Skip invalid boxes after clamping/padding
                if x1 <= x0 or y1 <= y0:
                    stats["filtered_invalid"] += 1
                    continue

                crop_img = im
                if args.use_masks:
                    mask = create_mask_from_segmentation(ann.get("segmentation"), im.size)
                    if mask is not None:
                        background = Image.new("RGB", im.size, (0, 0, 0))
                        crop_img = Image.composite(im, background, mask)

                crop = crop_img.crop((x0, y0, x1, y1))

                out_path = out_dir / split / super_name
                file_name = f"{img_id}_{ann.get('id')}.jpg"
                crop.save(out_path / file_name, format="JPEG", quality=90)

                stats["kept"] += 1
                stats["per_super"][super_name] += 1

    # Save stats
    total = stats["total_annotations"]
    stats["keep_rate"] = round(stats["kept"] / total, 4) if total else 0.0
    stats["filtered_small_rate"] = round(stats["filtered_small"] / total, 4) if total else 0.0

    area_ratios_sorted = sorted(area_ratios)
    stats["area_ratio_quantiles"] = {
        "p10": quantile(area_ratios_sorted, 0.10),
        "p25": quantile(area_ratios_sorted, 0.25),
        "p50": quantile(area_ratios_sorted, 0.50),
        "p75": quantile(area_ratios_sorted, 0.75),
        "p90": quantile(area_ratios_sorted, 0.90),
    }
    stats["suggest_keep"] = args.suggest_keep
    stats["suggested_min_box_area_ratio"] = quantile(area_ratios_sorted, 1.0 - args.suggest_keep)

    # Optional histogram
    hist_path = args.hist_path or (out_dir / "area_ratio_hist.png")
    if area_ratios_sorted:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(area_ratios_sorted, bins=50, color="#4e79a7", alpha=0.85)
        ax.set_title("BBox area ratio distribution")
        ax.set_xlabel("bbox_area / image_area")
        ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(hist_path, dpi=200)
        plt.close(fig)

    stats["per_super"] = dict(stats["per_super"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved dataset to {out_dir}")
    print(f"Saved label map to {args.label_map}")


if __name__ == "__main__":
    main()
