"""
Train Tier-1 classifier from split CSVs (config-driven).
Usage:
  python CNN/clf_7cats_tier1/train.py
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import timm
import torch
import torch.nn as nn
import yaml
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except Exception:
    pillow_heif = None

DEFAULT_LABELS = ["paper", "plastic", "metal", "glass", "other_uncertain", "e-waste", "textile"]

MODEL_ALIASES = {
    "efficientnet_b0": "efficientnet_b0",
    "mobilenet_v3_large": "mobilenetv3_large_100",
    "mobilenetv3_large_100": "mobilenetv3_large_100",
}
LOGGER = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CsvDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        label_to_idx: Dict[str, int],
        transform=None,
        verify_images: bool = False,
        skip_bad_images: bool = True,
        max_retry: int = 5,
        bad_list_path: Path | None = None,
    ) -> None:
        self.csv_path = csv_path
        self.label_to_idx = label_to_idx
        self.transform = transform
        self.verify_images = verify_images
        self.skip_bad_images = skip_bad_images
        self.max_retry = max_retry
        self.bad_list_path = bad_list_path
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

        if self.verify_images and self.items:
            valid_items: List[Tuple[Path, int]] = []
            for path, label in self.items:
                if self._is_image_valid(path):
                    valid_items.append((path, label))
                else:
                    self._log_bad(path)
            self.items = valid_items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if not self.items:
            raise ValueError("Dataset is empty after filtering bad images.")
        for _ in range(max(1, self.max_retry)):
            path, label = self.items[idx]
            try:
                image = Image.open(path).convert("RGBA").convert("RGB")
                if self.transform:
                    image = self.transform(image)
                return image, label
            except Exception:
                self._log_bad(path)
                if not self.skip_bad_images:
                    raise
                idx = random.randint(0, len(self.items) - 1)
        raise ValueError("Failed to load image after retries.")

    def _is_image_valid(self, path: Path) -> bool:
        try:
            with Image.open(path) as img:
                img.verify()
            with Image.open(path) as img:
                img.convert("RGB")
            return True
        except Exception:
            return False

    def _log_bad(self, path: Path) -> None:
        if not self.bad_list_path:
            return
        try:
            self.bad_list_path.parent.mkdir(parents=True, exist_ok=True)
            with self.bad_list_path.open("a", encoding="utf-8") as f:
                f.write(str(path) + "\n")
        except OSError as exc:
            LOGGER.warning(
                "Failed to append bad image path to %s: %s",
                self.bad_list_path,
                exc,
            )


def build_transforms(img_size: int, mean: List[float], std: List[float], aug_cfg: dict):
    scale_min = float(aug_cfg.get("scale_min", 0.6))
    color_jitter = aug_cfg.get("color_jitter", [0.25, 0.25, 0.25, 0.1])
    perspective_distortion = float(aug_cfg.get("perspective_distortion", 0.4))
    perspective_p = float(aug_cfg.get("perspective_p", 0.4))
    blur_kernel = int(aug_cfg.get("blur_kernel", 3))
    blur_sigma = aug_cfg.get("blur_sigma", [0.1, 2.0])
    random_erasing_p = float(aug_cfg.get("random_erasing_p", 0.25))

    train_tfm = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(scale_min, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(*color_jitter),
            transforms.RandomPerspective(distortion_scale=perspective_distortion, p=perspective_p),
            transforms.GaussianBlur(kernel_size=blur_kernel, sigma=tuple(blur_sigma)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=random_erasing_p, value="random"),
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
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    topk: int = 3,
) -> Tuple[Dict[str, float], List[int], List[int]]:
    model.eval()
    all_labels: List[int] = []
    all_preds: List[int] = []
    topk_correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        top1 = torch.argmax(probs, dim=1)
        topk_preds = torch.topk(probs, k=min(topk, num_classes), dim=1).indices
        topk_correct += (topk_preds == labels.unsqueeze(1)).any(dim=1).sum().item()

        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(top1.cpu().tolist())
        total += labels.size(0)

    if total == 0:
        return {"top1": 0.0, "top3": 0.0, "macro_f1": 0.0}, [], []

    top1_acc = sum(int(p == y) for p, y in zip(all_preds, all_labels)) / total
    top3_acc = topk_correct / total
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return {"top1": top1_acc, "top3": top3_acc, "macro_f1": macro_f1}, all_labels, all_preds


def save_confusion_from_preds(labels: List[int], preds: List[int], class_names: List[str], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    if not labels:
        return
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
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


def save_history(history: List[Dict[str, float]], run_dir: Path) -> None:
    if not history:
        return
    json_path = run_dir / "training_history.json"
    csv_path = run_dir / "training_history.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    try:
        import matplotlib.pyplot as plt

        epochs = [h["epoch"] for h in history]
        train_loss = [h["train_loss"] for h in history]
        val_top1 = [h["val_top1"] for h in history]
        val_f1 = [h["val_f1"] for h in history]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
        ax1.plot(epochs, train_loss, label="train_loss")
        ax1.set_ylabel("Loss")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.plot(epochs, val_top1, label="val_top1")
        ax2.plot(epochs, val_f1, label="val_f1")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Score")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        fig.tight_layout()
        fig.savefig(run_dir / "training_curves.png", dpi=150)
        plt.close(fig)
    except Exception as exc:
        LOGGER.warning(
            "Failed to render training curves at %s: %s",
            run_dir / "training_curves.png",
            exc,
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Tier-1 classifier.")
    p.add_argument(
        "--config",
        type=Path,
        default=Path("CNN/clf_7cats_tier1/configs/train.yaml"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("train.yaml must be a YAML mapping.")

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    labels = cfg.get("labels", DEFAULT_LABELS)
    if not labels:
        raise ValueError("labels is empty in train.yaml.")

    device_str = cfg.get("device", "cuda")
    require_cuda = bool(cfg.get("require_cuda", False))
    if device_str == "cuda" and not torch.cuda.is_available():
        if require_cuda:
            raise RuntimeError("require_cuda=true but CUDA is not available.")
        device = torch.device("cpu")
    else:
        device = torch.device(device_str)

    paths = cfg.get("paths", {})
    train_csv = Path(paths.get("train_csv", "CNN/data/tier1_splits/train.csv"))
    val_csv = Path(paths.get("val_csv", "CNN/data/tier1_splits/val.csv"))
    test_csv = Path(paths.get("test_csv", "CNN/data/tier1_splits/test.csv"))
    output_dir = Path(paths.get("output_dir", "CNN/outputs/clf_7cats_tier1"))
    model_dir = Path(paths.get("model_dir", "CNN/models"))

    model_cfg = cfg.get("model", {})
    backbone = model_cfg.get("backbone", "efficientnet_b0")
    backbone = MODEL_ALIASES.get(backbone, backbone)
    pretrained = bool(model_cfg.get("pretrained", True))
    img_size = int(model_cfg.get("img_size", 224))

    data_cfg = cfg.get("data", {})
    batch_size = int(data_cfg.get("batch_size", 32))
    num_workers = int(data_cfg.get("num_workers", 4))
    pin_memory = bool(data_cfg.get("pin_memory", device.type == "cuda"))
    persistent_workers = bool(data_cfg.get("persistent_workers", num_workers > 0))
    prefetch_factor = int(data_cfg.get("prefetch_factor", 2))
    verify_images = bool(data_cfg.get("verify_images", False))
    skip_bad_images = bool(data_cfg.get("skip_bad_images", True))
    max_retry = int(data_cfg.get("max_retry", 5))

    training_cfg = cfg.get("training", {})
    freeze_epochs = int(training_cfg.get("freeze_epochs", 5))
    finetune_epochs = int(training_cfg.get("finetune_epochs", 15))
    grad_clip = float(training_cfg.get("grad_clip", 1.0))
    use_amp = bool(training_cfg.get("use_amp", True)) and device.type == "cuda"
    early_stop_patience = int(training_cfg.get("early_stop_patience", 0))
    show_progress = bool(training_cfg.get("show_progress", True))

    optim_cfg = cfg.get("optimizer", {})
    lr_freeze = float(optim_cfg.get("lr_freeze", 1e-3))
    lr_finetune = float(optim_cfg.get("lr_finetune", 1e-4))
    weight_decay = float(optim_cfg.get("weight_decay", 0.01))
    betas = tuple(optim_cfg.get("betas", [0.9, 0.999]))

    sched_cfg = cfg.get("scheduler", {})
    scheduler_enabled = bool(sched_cfg.get("enabled", True))
    scheduler_name = sched_cfg.get("name", "cosine")
    min_lr = float(sched_cfg.get("min_lr", 1e-6))

    loss_cfg = cfg.get("loss", {})
    label_smoothing = float(loss_cfg.get("label_smoothing", 0.05))

    infer_cfg = cfg.get("infer", {})

    print(f"Using device={device} | backbone={backbone} | pretrained={pretrained}")

    label_to_idx = {name: i for i, name in enumerate(labels)}

    model = timm.create_model(backbone, pretrained=pretrained, num_classes=len(labels))
    model.to(device)

    data_cfg_model = timm.data.resolve_data_config({}, model=model)
    mean = data_cfg_model["mean"]
    std = data_cfg_model["std"]

    aug_cfg = cfg.get("augment", {})
    train_tfm, eval_tfm = build_transforms(img_size, mean, std, aug_cfg)

    run_name = f"{backbone}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "config_snapshot.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    bad_list_path = run_dir / "bad_images.txt"

    train_ds = CsvDataset(
        train_csv,
        label_to_idx,
        transform=train_tfm,
        verify_images=verify_images,
        skip_bad_images=skip_bad_images,
        max_retry=max_retry,
        bad_list_path=bad_list_path,
    )
    val_ds = CsvDataset(
        val_csv,
        label_to_idx,
        transform=eval_tfm,
        verify_images=verify_images,
        skip_bad_images=skip_bad_images,
        max_retry=max_retry,
        bad_list_path=bad_list_path,
    )
    test_ds = CsvDataset(
        test_csv,
        label_to_idx,
        transform=eval_tfm,
        verify_images=verify_images,
        skip_bad_images=skip_bad_images,
        max_retry=max_retry,
        bad_list_path=bad_list_path,
    )

    if len(train_ds) == 0:
        raise ValueError("Empty train dataset. Check split CSVs.")

    class_counts = [0] * len(labels)
    for _, label in train_ds.items:
        class_counts[label] += 1
    class_weights = [1.0 / max(1, c) for c in class_counts]
    sample_weights = [class_weights[label] for _, label in train_ds.items]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers and num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(train_ds, sampler=sampler, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    best_f1 = -1.0
    best_path = run_dir / "best.pt"
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)
    history: List[Dict[str, float]] = []

    def make_scheduler(optimizer: torch.optim.Optimizer, epochs: int):
        if not scheduler_enabled:
            return None
        if scheduler_name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, epochs), eta_min=min_lr
            )
        return None

    for param in model.parameters():
        param.requires_grad = False
    for param in model.get_classifier().parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_freeze,
        weight_decay=weight_decay,
        betas=betas,
    )
    scheduler = make_scheduler(optimizer, freeze_epochs)

    def run_epoch(epoch_idx: int, total_epochs: int) -> Tuple[float, float]:
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        train_iter = train_loader
        if show_progress:
            train_iter = tqdm(
                train_loader,
                desc=f"Train {epoch_idx}/{total_epochs}",
                leave=False,
                ncols=100,
            )
        for images, labels_batch in train_iter:
            images = images.to(device)
            labels_batch = labels_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels_batch)
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * labels_batch.size(0)
            total += labels_batch.size(0)
            correct += (logits.argmax(dim=1) == labels_batch).sum().item()
        avg_loss = total_loss / max(1, total)
        acc = correct / max(1, total)
        print(f"Epoch {epoch_idx}/{total_epochs} | train_loss={avg_loss:.4f} train_acc={acc:.4f}")
        return avg_loss, acc

    stop_early = False
    no_improve = 0

    for epoch in range(1, freeze_epochs + 1):
        train_loss, train_acc = run_epoch(epoch, freeze_epochs + finetune_epochs)
        if scheduler:
            scheduler.step()
        val_metrics, val_labels, val_preds = collect_predictions(
            model, val_loader, device, len(labels), topk=int(infer_cfg.get("topk", 3))
        )
        print(
            f"val_top1={val_metrics['top1']:.4f} val_top3={val_metrics['top3']:.4f} val_f1={val_metrics['macro_f1']:.4f}"
        )
        history.append(
            {
                "epoch": epoch,
                "stage": 1,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_top1": val_metrics["top1"],
                "val_top3": val_metrics["top3"],
                "val_f1": val_metrics["macro_f1"],
            }
        )
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), best_path)
            save_confusion_from_preds(val_labels, val_preds, labels, run_dir / "val_confusion.png")
            print(f"Saved best checkpoint to {best_path}")
            no_improve = 0
        else:
            no_improve += 1
            if early_stop_patience > 0 and no_improve >= early_stop_patience:
                stop_early = True
                break

    if not stop_early:
        for param in model.parameters():
            param.requires_grad = True
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr_finetune,
            weight_decay=weight_decay,
            betas=betas,
        )
        scheduler = make_scheduler(optimizer, finetune_epochs)

        for epoch in range(freeze_epochs + 1, freeze_epochs + finetune_epochs + 1):
            train_loss, train_acc = run_epoch(epoch, freeze_epochs + finetune_epochs)
            if scheduler:
                scheduler.step()
            val_metrics, val_labels, val_preds = collect_predictions(
                model, val_loader, device, len(labels), topk=int(infer_cfg.get("topk", 3))
            )
            print(
                f"val_top1={val_metrics['top1']:.4f} val_top3={val_metrics['top3']:.4f} val_f1={val_metrics['macro_f1']:.4f}"
            )
            history.append(
                {
                    "epoch": epoch,
                    "stage": 2,
                    "lr": optimizer.param_groups[0]["lr"],
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_top1": val_metrics["top1"],
                    "val_top3": val_metrics["top3"],
                    "val_f1": val_metrics["macro_f1"],
                }
            )
            if val_metrics["macro_f1"] > best_f1:
                best_f1 = val_metrics["macro_f1"]
                torch.save(model.state_dict(), best_path)
                save_confusion_from_preds(val_labels, val_preds, labels, run_dir / "val_confusion.png")
                print(f"Saved best checkpoint to {best_path}")
                no_improve = 0
            else:
                no_improve += 1
                if early_stop_patience > 0 and no_improve >= early_stop_patience:
                    break

    save_history(history, run_dir)

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    test_metrics, test_labels, test_preds = collect_predictions(
        model, test_loader, device, len(labels), topk=int(infer_cfg.get("topk", 3))
    )
    save_confusion_from_preds(test_labels, test_preds, labels, run_dir / "test_confusion.png")
    with (run_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)
    report = classification_report(
        test_labels,
        test_preds,
        labels=list(range(len(labels))),
        target_names=labels,
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    with (run_dir / "test_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with (run_dir / "test_report.txt").open("w", encoding="utf-8") as f:
        f.write(
            classification_report(
                test_labels,
                test_preds,
                labels=list(range(len(labels))),
                target_names=labels,
                digits=4,
                zero_division=0,
            )
        )
    print(
        f"Test top1={test_metrics['top1']:.4f} top3={test_metrics['top3']:.4f} macro_f1={test_metrics['macro_f1']:.4f}"
    )

    model_path = model_dir / "tier1_best.pt"
    torch.save(model.state_dict(), model_path)

    label_map = {"labels": labels}
    with (model_dir / "label_map.json").open("w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)

    infer_config = {
        "backbone": backbone,
        "img_size": img_size,
        "mean": mean,
        "std": std,
        "topk": int(infer_cfg.get("topk", 3)),
        "conf_threshold": float(infer_cfg.get("conf_threshold", 0.75)),
        "margin_threshold": float(infer_cfg.get("margin_threshold", 0.15)),
    }
    with (model_dir / "infer_config.json").open("w", encoding="utf-8") as f:
        json.dump(infer_config, f, indent=2, ensure_ascii=False)

    print(f"Saved model -> {model_path}")


if __name__ == "__main__":
    main()
