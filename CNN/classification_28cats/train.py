import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data.adapters import ImageFolderAdapter
from src.data.taco_coco import load_label_map
from src.models.classifier import build_classifier
from src.utils import (
    ensure_dir,
    load_yaml,
    resolve_device,
    save_checkpoint,
    save_json,
    set_seed,
    snapshot_config,
    timestamp,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TACO super-28 classifier")
    p.add_argument("--config", type=Path, default=Path("ml/classification_28cats/configs/taco_super28.yaml"))
    return p.parse_args()


def topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    topk = torch.topk(logits, k=k, dim=1).indices
    correct = (topk == labels.unsqueeze(1)).any(dim=1).float()
    return float(correct.mean().item())


def run_epoch(model, loader, criterion, optimizer, device, grad_clip: float, train: bool) -> Dict[str, float]:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="train" if train else "val", leave=False):
        images, labels = images.to(device), labels.to(device)

        with torch.set_grad_enabled(train):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if train:
            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    if total == 0:
        raise ValueError("No samples found in the current split. Check dataset build/splits.")
    return {"loss": total_loss / total, "acc": total_correct / total}


@torch.no_grad()
def evaluate(model, loader, device, class_names: List[str], topk: int = 3) -> Dict[str, object]:
    model.eval()
    all_labels: List[int] = []
    all_preds: List[int] = []
    all_probs: List[List[float]] = []

    for images, labels in tqdm(loader, desc="test", leave=False):
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = probs.argmax(1)

        all_labels.extend(labels.tolist())
        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

    if not all_labels:
        raise ValueError("No samples found in test split. Check dataset build/splits.")
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    label_ids = list(range(len(class_names)))
    report_text = classification_report(
        all_labels,
        all_preds,
        labels=label_ids,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    report_dict = classification_report(
        all_labels,
        all_preds,
        labels=label_ids,
        target_names=class_names,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(all_labels, all_preds, labels=label_ids)

    # top-k accuracy
    probs_tensor = torch.tensor(all_probs)
    labels_tensor = torch.tensor(all_labels)
    topk_acc = topk_accuracy(probs_tensor, labels_tensor, k=min(topk, len(class_names)))

    return {
        "macro_f1": macro_f1,
        "report_text": report_text,
        "report_dict": report_dict,
        "confusion_matrix": cm,
        "topk_acc": topk_acc,
    }


def save_confusion_matrix(cm, class_names: List[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_training_plots(history: List[Dict[str, float]], out_dir: Path) -> None:
    epochs = [m["epoch"] for m in history]
    train_loss = [m["train_loss"] for m in history]
    val_loss = [m["val_loss"] for m in history]
    train_acc = [m["train_acc"] for m in history]
    val_acc = [m["val_acc"] for m in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, train_loss, label="train")
    axes[0].plot(epochs, val_loss, label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, train_acc, label="train")
    axes[1].plot(epochs, val_acc, label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    set_seed(cfg.get("seed", 42))
    device = resolve_device(cfg.get("device", "cpu"), cfg.get("require_cuda", False))
    print(f"Using device={device} | cuda_available={torch.cuda.is_available()} | cuda_version={torch.version.cuda}")

    adapter = ImageFolderAdapter(
        data_dir=cfg["data_dir"],
        backbone=cfg.get("backbone", "efficientnet_b0"),
        image_size=cfg.get("image_size", 224),
        batch_size=cfg.get("batch_size", 32),
        num_workers=cfg.get("num_workers", 4),
        persistent_workers=cfg.get("persistent_workers", True),
        prefetch_factor=cfg.get("prefetch_factor", 2),
        use_weighted_sampler=cfg.get("use_weighted_sampler", True),
    )

    loaders, class_weights, class_names = adapter.get_dataloaders(device)

    fine_to_super, super_classes = load_label_map(Path(cfg["label_map"]))
    missing = set(class_names) - set(super_classes)
    if missing:
        raise ValueError(f"Label map missing classes: {sorted(missing)}")
    missing_in_train = set(super_classes) - set(class_names)
    if missing_in_train:
        print(f"Warning: train split missing classes: {sorted(missing_in_train)}")

    model, backbone = build_classifier(
        num_classes=len(class_names),
        backbone=cfg.get("backbone", "efficientnet_b0"),
        pretrained=not cfg.get("no_pretrain", False),
    )
    model.to(device)

    weight_tensor = class_weights.to(device) if cfg.get("use_class_weights", True) else None
    criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=cfg.get("label_smoothing", 0.0))
    optimizer = AdamW(model.parameters(), lr=cfg.get("lr", 1e-3), weight_decay=cfg.get("weight_decay", 1e-4))
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.get("epochs", 20))

    run_name = cfg.get("run_name") or f"{backbone}_{timestamp()}"
    output_dir = Path(cfg.get("output_dir", "ml/outputs/classification_28cats")) / run_name
    artifact_dir = Path(cfg.get("artifact_dir", "ml/artifacts/classification_28cats")) / run_name
    ensure_dir(output_dir)
    ensure_dir(artifact_dir)
    snapshot_config(cfg, output_dir)

    best_acc = 0.0
    history: List[Dict[str, float]] = []

    for epoch in range(cfg.get("epochs", 20)):
        print(f"\nEpoch {epoch + 1}/{cfg.get('epochs', 20)}")
        train_metrics = run_epoch(
            model, loaders["train"], criterion, optimizer, device, cfg.get("grad_clip", 1.0), train=True
        )
        val_metrics = run_epoch(
            model, loaders["val"], criterion, optimizer, device, cfg.get("grad_clip", 1.0), train=False
        )
        scheduler.step()

        metrics = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(metrics)
        print(
            f"train_loss={metrics['train_loss']:.4f} train_acc={metrics['train_acc']:.4f} "
            f"val_loss={metrics['val_loss']:.4f} val_acc={metrics['val_acc']:.4f}"
        )

        state = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "class_names": class_names,
            "backbone": backbone,
            "image_size": cfg.get("image_size", 224),
            "timestamp": timestamp(),
            "config": cfg,
            "history": history,
        }
        save_checkpoint(state, artifact_dir / "last.pt")

        if val_metrics["acc"] >= best_acc:
            best_acc = val_metrics["acc"]
            save_checkpoint(state, artifact_dir / "best.pt")
            print(f"Saved new best checkpoint to {artifact_dir / 'best.pt'}")

    save_json({"history": history, "best_val_acc": best_acc}, output_dir / "metrics.json")
    save_training_plots(history, output_dir)

    # Final test evaluation using best checkpoint
    from src.utils import load_checkpoint  # local import to avoid circular

    best_state = load_checkpoint(artifact_dir / "best.pt", device)
    model.load_state_dict(best_state["model_state"])
    metrics = evaluate(model, loaders["test"], device, class_names, topk=cfg.get("topk", 3))

    save_json(
        {"macro_f1": metrics["macro_f1"], "top3_acc": metrics["topk_acc"], "report": metrics["report_dict"]},
        output_dir / "test_report.json",
    )
    with (output_dir / "test_report.txt").open("w", encoding="utf-8") as f:
        f.write(metrics["report_text"])
        f.write(f"\nTop-3 Accuracy: {metrics['topk_acc']:.4f}\n")
        f.write(f"Macro F1: {metrics['macro_f1']:.4f}\n")

    save_confusion_matrix(metrics["confusion_matrix"], class_names, output_dir / "test_confusion_matrix.png")

    print(f"Training complete. Best val acc={best_acc:.4f}")
    print(f"Test Top-3 acc={metrics['topk_acc']:.4f} | Test macro-F1={metrics['macro_f1']:.4f}")
    print(f"Run outputs: {output_dir}")


if __name__ == "__main__":
    main()
