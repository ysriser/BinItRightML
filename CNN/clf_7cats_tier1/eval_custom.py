"""
Run inference on a custom folder test set and export visual results.
Usage:
  python CNN/clf_7cats_tier1/eval_custom.py --data-dir <path_to_testset>
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import timm
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torchvision import transforms
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".heic", ".heif"}
LOGGER = logging.getLogger(__name__)

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except Exception:
    pillow_heif = None


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def list_images(data_dir: Path, labels: List[str]) -> List[Tuple[Path, str]]:
    items: List[Tuple[Path, str]] = []
    for label_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        label = label_dir.name
        if label not in labels:
            continue
        for file in sorted(label_dir.rglob("*")):
            if file.is_file() and file.suffix.lower() in IMAGE_EXTS:
                items.append((file, label))
    return items


def build_transform(img_size: int, mean: List[float], std: List[float]):
    return transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.10)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def draw_overlay(img: Image.Image, text_lines: List[str]) -> Image.Image:
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    padding = 6
    line_h = font.getbbox("Ag")[3] + 4
    max_w = max(draw.textlength(line, font=font) for line in text_lines)
    box_h = line_h * len(text_lines) + padding * 2
    box_w = int(max_w) + padding * 2
    draw.rectangle([0, 0, box_w, box_h], fill=(0, 0, 0, 160))
    y = padding
    for line in text_lines:
        draw.text((padding, y), line, fill=(255, 255, 255), font=font)
        y += line_h
    return img


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a custom test folder.")
    p.add_argument("--data-dir", type=Path, default=Path("CNN/data/G3_SGData"))
    p.add_argument("--model-dir", type=Path, default=Path("CNN/models"))
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--max-images", type=int, default=0, help="0 = no limit")
    p.add_argument("--save-images", action="store_true", default=False)
    p.add_argument("--no-save-images", action="store_false", dest="save_images")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    label_map = load_json(args.model_dir / "label_map.json")
    infer_cfg = load_json(args.model_dir / "infer_config.json")
    labels = label_map["labels"]
    label_to_idx = {name: i for i, name in enumerate(labels)}

    backbone = infer_cfg.get("backbone", "efficientnet_b0")
    img_size = int(infer_cfg.get("img_size", 224))
    mean = infer_cfg.get("mean", [0.485, 0.456, 0.406])
    std = infer_cfg.get("std", [0.229, 0.224, 0.225])
    topk = int(args.topk)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = timm.create_model(backbone, pretrained=False, num_classes=len(labels))
    state = torch.load(args.model_dir / "tier1_best.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    transform = build_transform(img_size, mean, std)

    items = list_images(args.data_dir, labels)
    if args.max_images and args.max_images > 0:
        items = items[: args.max_images]
    if not items:
        raise ValueError("No images found. Check data-dir and label folders.")

    run_name = datetime.now().strftime("custom_eval_%Y%m%d_%H%M%S")
    output_dir = args.output_dir or Path("CNN/outputs/clf_7cats_tier1") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "vis"
    if args.save_images:
        vis_dir.mkdir(parents=True, exist_ok=True)

    predictions: List[Dict[str, object]] = []
    misclassified: List[Dict[str, object]] = []
    all_labels: List[int] = []
    all_preds: List[int] = []

    for path, true_label in tqdm(items, desc="Inference", ncols=100):
        img = Image.open(path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)
        top_vals, top_idxs = torch.topk(probs, k=min(topk, len(labels)))
        top3 = [
            {"label": labels[idx], "p": float(val)}
            for idx, val in zip(top_idxs.tolist(), top_vals.tolist())
        ]
        pred_label = top3[0]["label"]
        pred_conf = top3[0]["p"]

        row = {
            "filepath": str(path),
            "true_label": true_label,
            "pred_label": pred_label,
            "pred_conf": pred_conf,
            "topk": top3,
        }
        predictions.append(row)

        all_labels.append(label_to_idx[true_label])
        all_preds.append(label_to_idx[pred_label])

        if pred_label != true_label:
            misclassified.append(row)

        if args.save_images:
            rel_dir = vis_dir / true_label
            rel_dir.mkdir(parents=True, exist_ok=True)
            overlay_lines = [
                f"true: {true_label}",
                f"pred: {pred_label} ({pred_conf:.2f})",
                "top3: " + ", ".join([f"{x['label']} {x['p']:.2f}" for x in top3]),
            ]
            out_img = draw_overlay(img.copy(), overlay_lines)
            out_path = rel_dir / path.name
            out_img.save(out_path)

    total = len(all_labels)
    top1 = sum(int(p == y) for p, y in zip(all_preds, all_labels)) / total
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    report = classification_report(
        all_labels,
        all_preds,
        labels=list(range(len(labels))),
        target_names=labels,
        digits=4,
        zero_division=0,
        output_dict=True,
    )

    metrics = {"top1": top1, "macro_f1": macro_f1, "total": total}
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with (output_dir / "predictions.json").open("w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    with (output_dir / "misclassified.json").open("w", encoding="utf-8") as f:
        json.dump(misclassified, f, indent=2, ensure_ascii=False)

    with (output_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with (output_dir / "report.txt").open("w", encoding="utf-8") as f:
        f.write(
            classification_report(
                all_labels,
                all_preds,
                labels=list(range(len(labels))),
                target_names=labels,
                digits=4,
                zero_division=0,
            )
        )

    try:
        import matplotlib.pyplot as plt

        cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(labels))))
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(output_dir / "confusion.png", dpi=150)
        plt.close(fig)
    except Exception as exc:
        LOGGER.warning(
            "Failed to render confusion matrix at %s: %s",
            output_dir / "confusion.png",
            exc,
        )

    print(f"Saved outputs -> {output_dir}")
    print(f"Top1={top1:.4f} | Macro-F1={macro_f1:.4f} | Total={total}")


if __name__ == "__main__":
    main()
