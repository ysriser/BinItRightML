import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm

from src.data.adapters import ImageFolderAdapter
from src.models.classifier import build_classifier
from src.utils import ensure_dir, load_checkpoint, load_yaml, resolve_device, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate TACO super-28 classifier")
    p.add_argument("--config", type=Path, default=Path("ml/classification_28cats/configs/taco_super28.yaml"))
    p.add_argument("--checkpoint", type=Path, required=True)
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, class_names: List[str], topk: int = 3):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

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

    probs_tensor = torch.tensor(all_probs)
    labels_tensor = torch.tensor(all_labels)
    topk_idx = torch.topk(probs_tensor, k=min(topk, len(class_names)), dim=1).indices
    topk_acc = float((topk_idx == labels_tensor.unsqueeze(1)).any(dim=1).float().mean().item())

    return report_text, report_dict, cm, macro_f1, topk_acc


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


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    device = resolve_device(cfg.get("device", "cpu"), cfg.get("require_cuda", False))

    ckpt = load_checkpoint(args.checkpoint, device=device)
    class_names = ckpt["class_names"]

    adapter = ImageFolderAdapter(
        data_dir=cfg["data_dir"],
        backbone=ckpt.get("backbone", cfg.get("backbone", "efficientnet_b0")),
        image_size=ckpt.get("image_size", cfg.get("image_size", 224)),
        batch_size=cfg.get("batch_size", 32),
        num_workers=cfg.get("num_workers", 4),
        persistent_workers=cfg.get("persistent_workers", True),
        prefetch_factor=cfg.get("prefetch_factor", 2),
        use_weighted_sampler=False,
    )
    loaders, _, _ = adapter.get_dataloaders(device)

    model, _ = build_classifier(len(class_names), backbone=ckpt.get("backbone", cfg.get("backbone", "efficientnet_b0")), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    run_name = args.checkpoint.parent.name
    output_dir = Path(cfg.get("output_dir", "ml/outputs/classification_28cats")) / run_name
    ensure_dir(output_dir)

    report_text, report_dict, cm, macro_f1, topk_acc = evaluate(
        model, loaders["test"], device, class_names, topk=cfg.get("topk", 3)
    )

    save_json(
        {"macro_f1": macro_f1, "top3_acc": topk_acc, "report": report_dict},
        output_dir / "test_report.json",
    )
    with (output_dir / "test_report.txt").open("w", encoding="utf-8") as f:
        f.write(report_text)
        f.write(f"\nTop-3 Accuracy: {topk_acc:.4f}\n")
        f.write(f"Macro F1: {macro_f1:.4f}\n")

    save_confusion_matrix(cm, class_names, output_dir / "test_confusion_matrix.png")
    print(report_text)
    print(f"\nTop-3 Accuracy: {topk_acc:.4f} | Macro F1: {macro_f1:.4f}")


if __name__ == "__main__":
    main()
