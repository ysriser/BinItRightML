"""
Train Tier-1 classifier from split CSVs.
Usage:
  python ml/train_tier1.py
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

FINAL_LABELS = ["paper", "plastic", "metal", "glass", "other_uncertain", "e-waste", "textile"]

MODEL_ALIASES = {
    "efficientnet_b0": "efficientnet_b0",
    "mobilenet_v3_large": "mobilenetv3_large_100",
    "mobilenetv3_large_100": "mobilenetv3_large_100",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CsvDataset(Dataset):
    def __init__(self, csv_path: Path, label_to_idx: Dict[str, int], transform=None) -> None:
        self.csv_path = csv_path
        self.label_to_idx = label_to_idx
        self.transform = transform
        self.items: List[Tuple[Path, int]] = []
        root = Path.cwd().resolve()

        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = row["final_label"]
                if label not in label_to_idx:
                    continue
                path = Path(row["filepath"])
                if not path.is_absolute():
                    path = root / path
                self.items.append((path, label_to_idx[label]))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.items[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def build_transforms(img_size: int, mean: List[float], std: List[float]):
    train_tfm = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.25, 0.25, 0.25, 0.1),
            transforms.RandomPerspective(distortion_scale=0.4, p=0.4),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.25, value="random"),
        ]
    )
    eval_tfm = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.15)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train_tfm, eval_tfm


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    topk: int = 3,
) -> Dict[str, float]:
    model.eval()
    all_labels: List[int] = []
    all_preds: List[int] = []
    all_topk: List[int] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        top1 = torch.argmax(probs, dim=1)
        topk_preds = torch.topk(probs, k=min(topk, num_classes), dim=1).indices
        correct_topk = (topk_preds == labels.unsqueeze(1)).any(dim=1).sum().item()

        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(top1.cpu().tolist())
        all_topk.append(correct_topk)

    total = len(all_labels)
    if total == 0:
        return {"top1": 0.0, "top3": 0.0, "macro_f1": 0.0}

    top1_acc = sum(int(p == y) for p, y in zip(all_preds, all_labels)) / total
    top3_acc = sum(all_topk) / total
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return {"top1": top1_acc, "top3": top3_acc, "macro_f1": macro_f1}


@torch.no_grad()
def confusion_png(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: List[str],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    model.eval()
    all_labels: List[int] = []
    all_preds: List[int] = []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        preds = torch.argmax(logits, dim=1)
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    if not all_labels:
        return

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Tier-1 classifier.")
    p.add_argument("--train-csv", type=Path, default=Path("ml/data/splits/train.csv"))
    p.add_argument("--val-csv", type=Path, default=Path("ml/data/splits/val.csv"))
    p.add_argument("--test-csv", type=Path, default=Path("ml/data/splits/test.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("ml/outputs/tier1"))
    p.add_argument("--model-dir", type=Path, default=Path("ml/models"))
    p.add_argument("--backbone", type=str, default="efficientnet_b0")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--freeze-epochs", type=int, default=5)
    p.add_argument("--finetune-epochs", type=int, default=15)
    p.add_argument("--lr-freeze", type=float, default=1e-3)
    p.add_argument("--lr-finetune", type=float, default=1e-4)
    p.add_argument("--conf-threshold", type=float, default=0.75)
    p.add_argument("--margin-threshold", type=float, default=0.15)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    backbone_name = MODEL_ALIASES.get(args.backbone, args.backbone)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device={device} | backbone={backbone_name}")

    label_to_idx = {name: i for i, name in enumerate(FINAL_LABELS)}

    model = timm.create_model(backbone_name, pretrained=True, num_classes=len(FINAL_LABELS))
    model.to(device)

    data_cfg = timm.data.resolve_data_config({}, model=model)
    mean = data_cfg["mean"]
    std = data_cfg["std"]

    train_tfm, eval_tfm = build_transforms(args.img_size, mean, std)

    train_ds = CsvDataset(args.train_csv, label_to_idx, transform=train_tfm)
    val_ds = CsvDataset(args.val_csv, label_to_idx, transform=eval_tfm)
    test_ds = CsvDataset(args.test_csv, label_to_idx, transform=eval_tfm)

    if len(train_ds) == 0:
        raise ValueError("Empty train dataset. Check split CSVs.")

    # WeightedRandomSampler for class balance.
    class_counts = [0] * len(FINAL_LABELS)
    for _, label in train_ds.items:
        class_counts[label] += 1
    class_weights = [1.0 / max(1, c) for c in class_counts]
    sample_weights = [class_weights[label] for _, label in train_ds.items]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    run_name = f"{backbone_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_f1 = -1.0
    best_path = run_dir / "best.pt"

    # Stage 1: freeze backbone.
    for param in model.parameters():
        param.requires_grad = False
    for param in model.get_classifier().parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_freeze)

    def run_epoch(epoch_idx: int, total_epochs: int) -> float:
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
        avg_loss = total_loss / max(1, total)
        acc = correct / max(1, total)
        print(f"Epoch {epoch_idx}/{total_epochs} | train_loss={avg_loss:.4f} train_acc={acc:.4f}")
        return avg_loss

    # Freeze stage.
    for epoch in range(1, args.freeze_epochs + 1):
        run_epoch(epoch, args.freeze_epochs + args.finetune_epochs)
        val_metrics = evaluate(model, val_loader, device, len(FINAL_LABELS))
        print(
            f"val_top1={val_metrics['top1']:.4f} val_top3={val_metrics['top3']:.4f} val_f1={val_metrics['macro_f1']:.4f}"
        )
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), best_path)
            confusion_png(model, val_loader, device, FINAL_LABELS, run_dir / "val_confusion.png")
            print(f"Saved best checkpoint to {best_path}")

    # Stage 2: unfreeze full model.
    for param in model.parameters():
        param.requires_grad = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_finetune)

    for epoch in range(args.freeze_epochs + 1, args.freeze_epochs + args.finetune_epochs + 1):
        run_epoch(epoch, args.freeze_epochs + args.finetune_epochs)
        val_metrics = evaluate(model, val_loader, device, len(FINAL_LABELS))
        print(
            f"val_top1={val_metrics['top1']:.4f} val_top3={val_metrics['top3']:.4f} val_f1={val_metrics['macro_f1']:.4f}"
        )
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), best_path)
            confusion_png(model, val_loader, device, FINAL_LABELS, run_dir / "val_confusion.png")
            print(f"Saved best checkpoint to {best_path}")

    # Load best and evaluate on test.
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_metrics = evaluate(model, test_loader, device, len(FINAL_LABELS))
    confusion_png(model, test_loader, device, FINAL_LABELS, run_dir / "test_confusion.png")
    with (run_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)
    print(
        f"Test top1={test_metrics['top1']:.4f} top3={test_metrics['top3']:.4f} macro_f1={test_metrics['macro_f1']:.4f}"
    )

    # Save best checkpoint and metadata to model dir.
    model_path = args.model_dir / "tier1_best.pt"
    torch.save(model.state_dict(), model_path)

    label_map = {"labels": FINAL_LABELS}
    with (args.model_dir / "label_map.json").open("w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)

    infer_config = {
        "backbone": backbone_name,
        "img_size": args.img_size,
        "mean": mean,
        "std": std,
        "topk": 3,
        "conf_threshold": args.conf_threshold,
        "margin_threshold": args.margin_threshold,
    }
    with (args.model_dir / "infer_config.json").open("w", encoding="utf-8") as f:
        json.dump(infer_config, f, indent=2, ensure_ascii=False)

    print(f"Saved model -> {model_path}")


if __name__ == "__main__":
    main()
