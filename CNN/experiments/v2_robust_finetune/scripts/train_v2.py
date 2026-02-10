"""Robust fine-tune v2 training for Tier-1 (7-class) classifier."""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import timm
import torch
import torch.nn as nn
import yaml
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except Exception:
    pillow_heif = None

try:
    import wandb
except Exception:
    wandb = None

warnings.filterwarnings(
    "ignore",
    message="Palette images with Transparency expressed in bytes should be converted",
    category=UserWarning,
)

IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".heic",
    ".heif",
}


@dataclass
class DomainSplit:
    train: List[Tuple[Path, int]]
    val: List[Tuple[Path, int]]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("train config must be a YAML mapping")
    return data


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def list_images_from_folder(
    data_dir: Path,
    label_to_idx: Dict[str, int],
) -> List[Tuple[Path, int]]:
    items: List[Tuple[Path, int]] = []
    for label_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        label = label_dir.name
        if label not in label_to_idx:
            continue
        for file in sorted(label_dir.rglob("*")):
            if file.is_file() and file.suffix.lower() in IMAGE_EXTS:
                items.append((file, label_to_idx[label]))
    return items


def list_images_from_manifest(
    manifest: Path,
    label_to_idx: Dict[str, int],
) -> List[Tuple[Path, int]]:
    items: List[Tuple[Path, int]] = []
    with manifest.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("Manifest must include headers")
        fieldnames = {name.lower(): name for name in reader.fieldnames}
        if "filepath" not in fieldnames:
            raise ValueError("Manifest must include 'filepath' column")
        label_field = (
            fieldnames.get("label")
            or fieldnames.get("class")
            or fieldnames.get("final_label")
        )
        if not label_field:
            raise ValueError("Manifest must include label/class/final_label column")
        for row in reader:
            raw_path = row[fieldnames["filepath"]]
            label = row[label_field]
            if label not in label_to_idx:
                continue
            path = Path(raw_path)
            if not path.is_absolute():
                path = manifest.parent / path
            items.append((path, label_to_idx[label]))
    return items


def build_domain_items(
    data_dir: Path,
    label_to_idx: Dict[str, int],
) -> List[Tuple[Path, int]]:
    manifest_candidates = [data_dir / "manifest.csv", data_dir / "labels.csv"]
    for candidate in manifest_candidates:
        if candidate.exists():
            return list_images_from_manifest(candidate, label_to_idx)
    return list_images_from_folder(data_dir, label_to_idx)


def split_items(
    items: List[Tuple[Path, int]],
    val_ratio: float,
    seed: int,
) -> DomainSplit:
    rng = random.Random(seed)
    shuffled = items[:]
    rng.shuffle(shuffled)
    val_size = int(round(len(shuffled) * val_ratio))
    val_items = shuffled[:val_size]
    train_items = shuffled[val_size:]
    return DomainSplit(train=train_items, val=val_items)


class CsvDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        label_to_idx: Dict[str, int],
        transform=None,
        verify_images: bool = False,
        skip_bad_images: bool = True,
        max_retry: int = 5,
        bad_list_path: Optional[Path] = None,
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
                label = row.get("final_label") or row.get("label")
                if label not in label_to_idx:
                    continue
                path = Path(row["filepath"])
                if not path.is_absolute():
                    path = root / path
                self.items.append((path, label_to_idx[label]))

    def __len__(self) -> int:
        return len(self.items)

    def _load_image(self, path: Path) -> Image.Image:
        with Image.open(path) as opened:
            img = opened.convert("RGB")
        return img

    def __getitem__(self, idx: int):
        for _ in range(max(1, self.max_retry)):
            path, label = self.items[idx]
            try:
                img = self._load_image(path)
                if self.verify_images:
                    img.verify()
                img = self._load_image(path)
                if self.transform:
                    img = self.transform(img)
                return img, label
            except Exception:
                if self.skip_bad_images:
                    idx = random.randint(0, len(self.items) - 1)
                    continue
                raise
        return self.__getitem__(random.randint(0, len(self.items) - 1))


class ListDataset(Dataset):
    def __init__(
        self,
        items: List[Tuple[Path, int]],
        transform=None,
        skip_bad_images: bool = True,
        max_retry: int = 5,
    ) -> None:
        self.items = items
        self.transform = transform
        self.skip_bad_images = skip_bad_images
        self.max_retry = max(1, max_retry)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        for _ in range(self.max_retry):
            path, label = self.items[idx]
            try:
                with Image.open(path) as opened:
                    img = opened.convert("RGB")
                if self.transform:
                    img = self.transform(img)
                return img, label
            except Exception:
                if not self.skip_bad_images:
                    raise
                idx = random.randint(0, len(self.items) - 1)
        return self.__getitem__(random.randint(0, len(self.items) - 1))


class ScaleDownPad:
    def __init__(
        self,
        img_size: int,
        scale_range: Tuple[float, float],
        prob: float,
        random_position: bool,
        pad_value: Tuple[int, int, int],
    ) -> None:
        self.img_size = img_size
        self.scale_range = scale_range
        self.prob = prob
        self.random_position = random_position
        self.pad_value = pad_value

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.prob:
            return img
        scale = random.uniform(*self.scale_range)
        new_size = max(1, int(round(self.img_size * scale)))
        resized = img.resize((new_size, new_size), resample=Image.BICUBIC)
        canvas = Image.new("RGB", (self.img_size, self.img_size), self.pad_value)
        if self.random_position:
            max_x = self.img_size - new_size
            max_y = self.img_size - new_size
            left = random.randint(0, max_x) if max_x > 0 else 0
            top = random.randint(0, max_y) if max_y > 0 else 0
        else:
            left = (self.img_size - new_size) // 2
            top = (self.img_size - new_size) // 2
        canvas.paste(resized, (left, top))
        return canvas


class RRCOrCenter:
    def __init__(
        self,
        rrc: transforms.RandomResizedCrop,
        img_size: int,
        rrc_p: float,
    ) -> None:
        self.rrc = rrc
        self.img_size = img_size
        self.rrc_p = rrc_p
        self._resize = transforms.Resize(
            int(img_size * 1.15),
            interpolation=InterpolationMode.BICUBIC,
        )
        self._center_crop = transforms.CenterCrop(img_size)

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.rrc_p:
            return self.rrc(img)
        return self._center_crop(self._resize(img))


def build_train_transform(
    cfg: dict,
    img_size: int,
    mean: Sequence[float],
    std: Sequence[float],
    phase: str,
):
    aug = cfg.get("augment", {})
    phase_cfg = cfg.get("phase_b", {}) if phase == "b" else {}
    phase_aug = phase_cfg.get("augment", {}) or {}

    rrc_p = float(aug.get("rrc_p", 1.0))
    rrc_p = float(phase_cfg.get("rrc_p", rrc_p))
    scale_min = float(aug.get("rrc_scale_min", 0.15))
    scale_max = float(aug.get("rrc_scale_max", 1.0))
    ratio = aug.get("rrc_ratio", [0.75, 1.33])
    color_jitter = phase_aug.get(
        "color_jitter", aug.get("color_jitter", [0.3, 0.3, 0.3, 0.1])
    )
    color_jitter_p = float(
        phase_aug.get("color_jitter_p", aug.get("color_jitter_p", 1.0))
    )
    hflip_p = float(aug.get("hflip_p", 0.5))

    scale_cfg = dict(aug.get("scale_down_pad", {}) or {})
    phase_scale_cfg = phase_aug.get("scale_down_pad")
    if isinstance(phase_scale_cfg, dict):
        scale_cfg.update(phase_scale_cfg)
    scale_enabled = bool(scale_cfg.get("enabled", True))

    affine_cfg = aug.get("affine", {}) or {}
    perspective_cfg = aug.get("perspective", {}) or {}
    blur_cfg = aug.get("blur", {}) or {}

    rrc = transforms.RandomResizedCrop(
        img_size,
        scale=(scale_min, scale_max),
        ratio=tuple(ratio),
        interpolation=InterpolationMode.BICUBIC,
    )
    transforms_list: List[transforms.Transform] = [
        RRCOrCenter(rrc=rrc, img_size=img_size, rrc_p=rrc_p)
    ]

    if scale_enabled:
        pad_mode = str(scale_cfg.get("pad_mode", "mean")).lower()
        if pad_mode == "mean":
            pad_value = tuple(int(round(m * 255)) for m in mean)
        else:
            pad_value = (0, 0, 0)
        transforms_list.append(
            ScaleDownPad(
                img_size=img_size,
                scale_range=tuple(scale_cfg.get("scale_range", [0.3, 0.7])),
                prob=float(scale_cfg.get("prob", 0.5)),
                random_position=bool(scale_cfg.get("random_position", True)),
                pad_value=pad_value,
            )
        )

    transforms_list.append(transforms.RandomHorizontalFlip(p=hflip_p))
    if color_jitter_p > 0:
        transforms_list.append(
            transforms.RandomApply(
                [transforms.ColorJitter(*color_jitter)],
                p=color_jitter_p,
            )
        )

    if float(affine_cfg.get("p", 0.0)) > 0:
        transforms_list.append(
            transforms.RandomApply(
                [
                    transforms.RandomAffine(
                        degrees=float(affine_cfg.get("degrees", 0.0)),
                        translate=tuple(affine_cfg.get("translate", [0.0, 0.0])),
                        scale=tuple(affine_cfg.get("scale", [1.0, 1.0])),
                        shear=tuple(affine_cfg.get("shear", [0.0, 0.0])),
                        interpolation=InterpolationMode.BICUBIC,
                    )
                ],
                p=float(affine_cfg.get("p", 0.0)),
            )
        )

    if float(perspective_cfg.get("p", 0.0)) > 0:
        transforms_list.append(
            transforms.RandomPerspective(
                distortion_scale=float(perspective_cfg.get("distortion", 0.3)),
                p=float(perspective_cfg.get("p", 0.2)),
                interpolation=InterpolationMode.BICUBIC,
            )
        )

    if bool(blur_cfg.get("enabled", False)):
        transforms_list.append(
            transforms.RandomApply(
                [
                    transforms.GaussianBlur(
                        kernel_size=int(blur_cfg.get("kernel", 3)),
                        sigma=tuple(blur_cfg.get("sigma", [0.1, 2.0])),
                    )
                ],
                p=float(blur_cfg.get("p", 0.2)),
            )
        )

    transforms_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    erasing_p = float(
        phase_aug.get("random_erasing_p", aug.get("random_erasing_p", 0.0))
    )
    erasing_scale = tuple(
        phase_aug.get(
            "random_erasing_scale", aug.get("random_erasing_scale", [0.02, 0.33])
        )
    )
    erasing_ratio = tuple(
        phase_aug.get(
            "random_erasing_ratio", aug.get("random_erasing_ratio", [0.3, 3.3])
        )
    )
    if erasing_p > 0:
        transforms_list.append(
            transforms.RandomErasing(
                p=erasing_p,
                scale=erasing_scale,
                ratio=erasing_ratio,
            )
        )

    return transforms.Compose(transforms_list)


def build_eval_transform(img_size: int, mean: Sequence[float], std: Sequence[float]):
    return transforms.Compose(
        [
            transforms.Resize(
                int(img_size * 1.15),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def compute_class_weights(
    items: List[Tuple[Path, int]],
    num_classes: int,
) -> torch.Tensor:
    counts = [0 for _ in range(num_classes)]
    for _, label in items:
        counts[label] += 1
    total = sum(counts)
    weights = [total / max(count, 1) for count in counts]
    return torch.tensor(weights, dtype=torch.float)


def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return torch.zeros(labels.size(0), num_classes, device=labels.device).scatter_(
        1, labels.unsqueeze(1), 1.0
    )


def soft_target_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=1)
    return -(targets * log_probs).sum(dim=1).mean()


class FocalCrossEntropy(nn.Module):
    """Focal loss wrapper for multi-class classification."""

    def __init__(
        self,
        gamma: float = 1.5,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(
            reduction="none",
            label_smoothing=label_smoothing,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        focal_weight = (1.0 - pt) ** self.gamma
        return (focal_weight * ce_loss).mean()

def rand_bbox(width: int, height: int, lam: float) -> Tuple[int, int, int, int]:
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)

    cx = np.random.randint(width)
    cy = np.random.randint(height)

    x1 = np.clip(cx - cut_w // 2, 0, width)
    y1 = np.clip(cy - cut_h // 2, 0, height)
    x2 = np.clip(cx + cut_w // 2, 0, width)
    y2 = np.clip(cy + cut_h // 2, 0, height)
    return x1, y1, x2, y2


def apply_mixup_cutmix(
    images: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    mixup_cfg: dict,
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    mixup_alpha = float(mixup_cfg.get("mixup_alpha", 0.0))
    mixup_p = float(mixup_cfg.get("mixup_p", 0.0))
    cutmix_alpha = float(mixup_cfg.get("cutmix_alpha", 0.0))
    cutmix_p = float(mixup_cfg.get("cutmix_p", 0.0))

    use_mixup = mixup_alpha > 0 and random.random() < mixup_p
    use_cutmix = cutmix_alpha > 0 and random.random() < cutmix_p

    if not use_mixup and not use_cutmix:
        return images, labels, False

    if use_mixup and use_cutmix:
        use_mixup = random.random() < 0.5
        use_cutmix = not use_mixup

    batch_size, _, height, width = images.size()
    perm = torch.randperm(batch_size).to(images.device)

    if use_mixup:
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        mixed = lam * images + (1 - lam) * images[perm]
        targets = one_hot(labels, num_classes)
        targets_perm = one_hot(labels[perm], num_classes)
        mixed_targets = lam * targets + (1 - lam) * targets_perm
        return mixed, mixed_targets, True

    lam = np.random.beta(cutmix_alpha, cutmix_alpha)
    x1, y1, x2, y2 = rand_bbox(width, height, lam)
    images[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
    lam_adjusted = 1 - ((x2 - x1) * (y2 - y1) / (width * height))
    targets = one_hot(labels, num_classes)
    targets_perm = one_hot(labels[perm], num_classes)
    mixed_targets = lam_adjusted * targets + (1 - lam_adjusted) * targets_perm
    return images, mixed_targets, True


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: List[str],
    topk: int = 3,
) -> Tuple[Dict[str, float], List[int], List[int]]:
    model.eval()
    all_labels: List[int] = []
    all_preds: List[int] = []
    total = 0
    top1_correct = 0
    topk_correct = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            total += labels.size(0)
            top1_correct += (preds == labels).sum().item()

            top_vals, top_idxs = torch.topk(probs, k=min(topk, probs.size(1)), dim=1)
            for i in range(labels.size(0)):
                if labels[i].item() in top_idxs[i].tolist():
                    topk_correct += 1

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    if total == 0:
        return {"top1": 0.0, "top3": 0.0, "macro_f1": 0.0}, [], []

    top1_acc = top1_correct / total
    top3_acc = topk_correct / total
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return (
        {"top1": top1_acc, "top3": top3_acc, "macro_f1": macro_f1},
        all_labels,
        all_preds,
    )


def save_confusion(
    labels: List[int],
    preds: List[int],
    class_names: List[str],
    out_path: Path,
) -> None:
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    try:
        import matplotlib.pyplot as plt

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
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    except Exception:
        pass


def save_history(
    rows: List[Dict[str, float]],
    out_dir: Path,
) -> None:
    if not rows:
        return
    rows = sorted(rows, key=lambda r: r.get("global_epoch", r.get("epoch", 0)))
    csv_path = out_dir / "metrics.csv"
    json_path = out_dir / "metrics.json"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    try:
        import matplotlib.pyplot as plt

        epochs = [
            row.get("global_epoch", row.get("epoch", 0)) for row in rows
        ]
        train_loss = [row["train_loss"] for row in rows]
        val_top1 = [row["val_top1"] for row in rows]
        val_f1 = [row["val_f1"] for row in rows]
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(epochs, train_loss, label="train_loss")
        ax[0].set_title("Train Loss")
        ax[0].set_xlabel("epoch")
        ax[0].legend()
        ax[1].plot(epochs, val_top1, label="val_top1")
        ax[1].plot(epochs, val_f1, label="val_f1")
        ax[1].set_title("Val Metrics")
        ax[1].set_xlabel("epoch")
        ax[1].legend()
        fig.tight_layout()
        fig.savefig(out_dir / "curves.png", dpi=150)
        plt.close(fig)
    except Exception:
        pass


def summarize_phases(history: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for phase_id in (1, 2):
        phase_rows = [row for row in history if int(row.get("phase", 0)) == phase_id]
        if not phase_rows:
            continue
        best = max(phase_rows, key=lambda r: r.get("val_f1", 0.0))
        summary[f"phase_{phase_id}"] = {
            "epoch": int(best.get("epoch", 0)),
            "global_epoch": int(best.get("global_epoch", best.get("epoch", 0))),
            "val_top1": float(best.get("val_top1", 0.0)),
            "val_top3": float(best.get("val_top3", 0.0)),
            "val_f1": float(best.get("val_f1", 0.0)),
            "train_loss": float(best.get("train_loss", 0.0)),
            "train_acc": float(best.get("train_acc", 0.0)),
        }
    return summary


def get_git_commit(repo_root: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        return None
    return None


def resolve_head_module(model: torch.nn.Module) -> Optional[torch.nn.Module]:
    if hasattr(model, "get_classifier"):
        head = model.get_classifier()
        if isinstance(head, torch.nn.Module):
            return head
    for attr in ("classifier", "fc", "head"):
        if hasattr(model, attr):
            head = getattr(model, attr)
            if isinstance(head, torch.nn.Module):
                return head
    return None


def setup_wandb(cfg: dict, run_dir: Path, run_name: str) -> Optional["wandb.sdk.wandb_run.Run"]:
    wb_cfg = cfg.get("wandb", {}) or {}
    if not wb_cfg.get("enabled", False):
        return None
    if wandb is None:
        raise ImportError("wandb is enabled in config but not installed.")
    project = wb_cfg.get("project", "bin-it-right")
    entity = wb_cfg.get("entity")
    tags = wb_cfg.get("tags", [])
    mode = wb_cfg.get("mode", "online")
    return wandb.init(
        project=project,
        entity=entity,
        tags=tags,
        mode=mode,
        name=run_name,
        dir=str(run_dir),
        config=cfg,
    )


def log_wandb(
    run: Optional["wandb.sdk.wandb_run.Run"],
    data: Dict[str, float],
) -> None:
    if run is None:
        return
    run.log(data)


def split_backbone_head_params(
    model: torch.nn.Module,
) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    head = resolve_head_module(model)
    if head is None:
        return list(model.parameters()), []
    head_ids = {id(p) for p in head.parameters()}
    head_params = [p for p in model.parameters() if id(p) in head_ids]
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    return backbone_params, head_params


def set_requires_grad(params: List[torch.nn.Parameter], value: bool) -> None:
    for param in params:
        param.requires_grad = value


def export_onnx_model(
    model: torch.nn.Module,
    artifact_dir: Path,
    labels: List[str],
    img_size: int,
    mean: Sequence[float],
    std: Sequence[float],
    opset: int,
    backbone: str,
    infer_cfg: dict,
) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = artifact_dir / "model.onnx"
    orig_device = next(model.parameters()).device
    model_cpu = model.to("cpu")
    model_cpu.eval()
    dummy = torch.randn(1, 3, img_size, img_size)
    torch.onnx.export(
        model_cpu,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
    )
    model.to(orig_device)
    infer_out = {
        "backbone": backbone,
        "img_size": img_size,
        "mean": mean,
        "std": std,
        "topk": infer_cfg.get("topk", 3),
        "conf_threshold": infer_cfg.get("conf_threshold", 0.75),
        "margin_threshold": infer_cfg.get("margin_threshold", 0.15),
    }
    save_json(artifact_dir / "infer_config.json", infer_out)
    save_json(artifact_dir / "label_map.json", {"labels": labels})
    return onnx_path


def update_models_dir(
    artifact_dir: Path,
    models_dir: Path,
) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "best.pt": "tier1_best.pt",
        "model.onnx": "tier1.onnx",
        "infer_config.json": "infer_config.json",
        "label_map.json": "label_map.json",
    }
    for src_name, dst_name in mapping.items():
        src = artifact_dir / src_name
        if src.exists():
            (models_dir / dst_name).write_bytes(src.read_bytes())


def save_phase_a_latest(
    model: torch.nn.Module,
    artifact_dir: Path,
    models_dir: Path,
) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    phase_a_path = artifact_dir / "phase_a_latest.pt"
    torch.save(model.state_dict(), phase_a_path)
    (models_dir / "tier1_phase_a_latest.pt").write_bytes(
        phase_a_path.read_bytes()
    )
    return phase_a_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Robust fine-tune v2")
    p.add_argument(
        "--config",
        type=Path,
        default=Path("CNN/experiments/v2_robust_finetune/configs/train_v2.yaml"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    device_name = cfg.get("device", "cuda")
    require_cuda = bool(cfg.get("require_cuda", False))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cpu":
        device = torch.device("cpu")
    if require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA required but not available")

    labels = cfg.get("labels", [])
    if not labels:
        raise ValueError("labels is empty in train config")
    label_to_idx = {name: i for i, name in enumerate(labels)}

    model_cfg = cfg.get("model", {})
    backbone = model_cfg.get("backbone", "efficientnet_b0")
    pretrained = bool(model_cfg.get("pretrained", True))
    img_size = int(model_cfg.get("img_size", 224))

    model = timm.create_model(backbone, pretrained=pretrained, num_classes=len(labels))
    model.to(device)

    data_cfg_model = timm.data.resolve_data_config({}, model=model)
    mean = data_cfg_model["mean"]
    std = data_cfg_model["std"]

    train_transform_a = build_train_transform(cfg, img_size, mean, std, phase="a")
    train_transform_b = build_train_transform(cfg, img_size, mean, std, phase="b")
    eval_transform = build_eval_transform(img_size, mean, std)

    data_cfg = cfg.get("data", {})
    base_batch_size = int(data_cfg.get("batch_size", 32))
    base_workers = int(data_cfg.get("num_workers", 4))
    base_pin = bool(data_cfg.get("pin_memory", True))
    base_persistent = bool(data_cfg.get("persistent_workers", True))
    base_prefetch = int(data_cfg.get("prefetch_factor", 2))

    def make_loader_kwargs(num_workers: int) -> Dict[str, object]:
        kwargs: Dict[str, object] = {
            "batch_size": base_batch_size,
            "num_workers": num_workers,
            "pin_memory": base_pin,
            "worker_init_fn": seed_worker if num_workers > 0 else None,
        }
        if num_workers > 0:
            kwargs["persistent_workers"] = base_persistent
            kwargs["prefetch_factor"] = base_prefetch
        return kwargs

    train_loader_kwargs = make_loader_kwargs(base_workers)
    eval_loader_kwargs = make_loader_kwargs(0)

    paths = cfg.get("paths", {})
    train_csv = Path(
        paths.get("train_csv", "CNN/data/tier1_splits/latest/train.csv")
    )
    val_csv = Path(paths.get("val_csv", "CNN/data/tier1_splits/latest/val.csv"))
    test_csv = Path(
        paths.get("test_csv", "CNN/data/tier1_splits/latest/test.csv")
    )
    g3_data_dir = Path(paths.get("g3_data_dir", "CNN/data/hardset"))

    output_root = Path(
        paths.get(
            "output_dir",
            "CNN/experiments/v2_robust_finetune/outputs",
        )
    )
    artifact_root = Path(
        paths.get(
            "artifact_dir",
            "CNN/experiments/v2_robust_finetune/artifacts",
        )
    )
    models_dir = Path(paths.get("models_dir", "CNN/models"))

    run_name = datetime.now().strftime("robust_v2_%Y%m%d_%H%M%S")
    run_dir = output_root / run_name
    artifact_dir = artifact_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "config_snapshot.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    wb_run = setup_wandb(cfg, run_dir, run_name)
    if wb_run is not None:
        wb_run.summary["train_csv"] = str(train_csv)
        wb_run.summary["val_csv"] = str(val_csv)
        wb_run.summary["test_csv"] = str(test_csv)
        wb_run.summary["domain_dir"] = str(g3_data_dir)

    phase_a = cfg.get("phase_a", {})
    phase_b = cfg.get("phase_b", {})
    phase_b_only = bool(phase_b.get("only", False))
    phase_a_epochs = int(phase_a.get("epochs", 10))
    phase_b_epochs = int(phase_b.get("epochs", 5))
    if phase_b_only:
        phase_a_epochs = 0

    train_ds = CsvDataset(
        train_csv,
        label_to_idx,
        transform=train_transform_a,
        verify_images=bool(data_cfg.get("verify_images", False)),
        skip_bad_images=bool(data_cfg.get("skip_bad_images", True)),
        max_retry=int(data_cfg.get("max_retry", 5)),
    )
    val_ds = CsvDataset(
        val_csv,
        label_to_idx,
        transform=eval_transform,
        verify_images=False,
        skip_bad_images=True,
        max_retry=1,
    )
    test_ds = CsvDataset(
        test_csv,
        label_to_idx,
        transform=eval_transform,
        verify_images=False,
        skip_bad_images=True,
        max_retry=1,
    )

    class_weights = compute_class_weights(train_ds.items, len(labels))
    sample_weights = [class_weights[label] for _, label in train_ds.items]
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(train_ds, sampler=sampler, **train_loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **eval_loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **eval_loader_kwargs)

    if not g3_data_dir.exists():
        raise FileNotFoundError(f"Missing domain data dir: {g3_data_dir}")
    domain_items = build_domain_items(g3_data_dir, label_to_idx)
    if not domain_items:
        raise ValueError("No domain images found in hardset")
    domain_split = split_items(
        domain_items,
        float(phase_b.get("val_ratio", 0.2)),
        seed,
    )
    domain_train_ds = ListDataset(
        domain_split.train,
        transform=train_transform_b,
        skip_bad_images=bool(data_cfg.get("skip_bad_images", True)),
        max_retry=int(data_cfg.get("max_retry", 5)),
    )
    domain_val_ds = ListDataset(
        domain_split.val,
        transform=eval_transform,
        skip_bad_images=bool(data_cfg.get("skip_bad_images", True)),
        max_retry=int(data_cfg.get("max_retry", 5)),
    )
    use_domain_val = bool(phase_b.get("prefer_domain_val", True)) and len(
        domain_val_ds
    ) > 0

    domain_class_weights = compute_class_weights(domain_train_ds.items, len(labels))
    domain_sample_weights = [
        domain_class_weights[label] for _, label in domain_train_ds.items
    ]
    domain_sampler = WeightedRandomSampler(
        domain_sample_weights,
        num_samples=len(domain_sample_weights),
        replacement=True,
    )
    domain_train_loader = DataLoader(
        domain_train_ds,
        sampler=domain_sampler,
        **train_loader_kwargs,
    )
    domain_val_loader = DataLoader(domain_val_ds, shuffle=False, **eval_loader_kwargs)

    phase_a_mix_cfg = phase_a.get("domain_mix", {}) or {}
    use_phase_a_domain_mix = (
        bool(phase_a_mix_cfg.get("enabled", False))
        and not phase_b_only
        and len(domain_split.train) > 0
    )
    if use_phase_a_domain_mix:
        repeat = max(1, int(phase_a_mix_cfg.get("repeat", 2)))
        mixed_items = list(train_ds.items) + (list(domain_split.train) * repeat)
        phase_a_train_ds = ListDataset(
            mixed_items,
            transform=train_transform_a,
            skip_bad_images=bool(data_cfg.get("skip_bad_images", True)),
            max_retry=int(data_cfg.get("max_retry", 5)),
        )
        mixed_weights = compute_class_weights(phase_a_train_ds.items, len(labels))
        mixed_sample_weights = [
            mixed_weights[label] for _, label in phase_a_train_ds.items
        ]
        mixed_sampler = WeightedRandomSampler(
            mixed_sample_weights,
            num_samples=len(mixed_sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            phase_a_train_ds,
            sampler=mixed_sampler,
            **train_loader_kwargs,
        )
        print(
            "Phase A domain mix enabled -> "
            f"base={len(train_ds.items)} domain={len(domain_split.train)} repeat={repeat} "
            f"total={len(phase_a_train_ds.items)}"
        )

    loss_cfg = cfg.get("loss", {})
    label_smoothing = float(loss_cfg.get("label_smoothing", 0.05))
    focal_cfg = loss_cfg.get("focal", {}) or {}
    use_focal = bool(focal_cfg.get("enabled", False))
    focal_gamma = float(focal_cfg.get("gamma", 1.5))

    ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    focal_loss = FocalCrossEntropy(
        gamma=focal_gamma,
        label_smoothing=label_smoothing,
    )

    mix_cfg_a = cfg.get("mixup_cutmix", {}) or {}
    mix_cfg_b = cfg.get("phase_b", {}).get("mixup_cutmix", mix_cfg_a) or {}
    scale_cfg = cfg.get("augment", {}).get("scale_down_pad", {}) or {}
    print(
        "Augmentations -> "
        f"scale_down_pad={scale_cfg.get('enabled', True)} "
        f"mixup_p={mix_cfg_a.get('mixup_p', 0.0)} "
        f"cutmix_p={mix_cfg_a.get('cutmix_p', 0.0)} "
        f"focal={use_focal} gamma={focal_gamma}"
    )

    optim_cfg = cfg.get("optimizer", {})
    weight_decay = float(optim_cfg.get("weight_decay", 0.01))
    betas = tuple(optim_cfg.get("betas", [0.9, 0.999]))

    def make_optimizer(lr: float):
        if optim_cfg.get("name", "adamw").lower() == "adamw":
            return torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=betas,
            )
        return torch.optim.Adam(model.parameters(), lr=lr)

    def make_optimizer_split(
        backbone_params: List[torch.nn.Parameter],
        head_params: List[torch.nn.Parameter],
        backbone_lr: float,
        head_lr: float,
    ) -> torch.optim.Optimizer:
        params = []
        if backbone_params:
            params.append(
                {
                    "params": backbone_params,
                    "lr": backbone_lr,
                    "weight_decay": weight_decay,
                }
            )
        if head_params:
            params.append(
                {
                    "params": head_params,
                    "lr": head_lr,
                    "weight_decay": weight_decay,
                }
            )
        if optim_cfg.get("name", "adamw").lower() == "adamw":
            return torch.optim.AdamW(params, betas=betas)
        return torch.optim.Adam(params)

    sched_cfg = cfg.get("scheduler", {})

    def make_scheduler(
        optimizer: torch.optim.Optimizer,
        epochs: int,
        override: Optional[dict] = None,
    ):
        cfg_local = dict(sched_cfg)
        if isinstance(override, dict):
            cfg_local.update(override)
        enabled = bool(cfg_local.get("enabled", True))
        if not enabled:
            return None
        name = cfg_local.get("name", "cosine")
        min_lr = float(cfg_local.get("min_lr", 1e-6))
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs,
                eta_min=min_lr,
            )
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, epochs // 3),
            gamma=0.1,
        )

    train_cfg = cfg.get("training", {})
    use_amp = bool(train_cfg.get("use_amp", True)) and device.type == "cuda"
    grad_clip = float(train_cfg.get("grad_clip", 0.0))
    early_patience = int(train_cfg.get("early_stop_patience", 0))
    show_progress = bool(train_cfg.get("show_progress", True))

    best_f1 = -1.0
    best_state: Optional[dict] = None
    best_metrics: Dict[str, float] = {}
    best_epoch = 0
    best_source = "val"
    no_improve = 0
    history: List[Dict[str, float]] = []

    def run_epoch(
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        mix_cfg: dict,
    ) -> Tuple[float, float]:
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        iterator = loader
        if show_progress:
            iterator = tqdm(loader, desc="train", ncols=100)

        for images, labels_batch in iterator:
            images = images.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                mixed_images, mixed_targets, mixed = apply_mixup_cutmix(
                    images,
                    labels_batch,
                    num_classes=len(labels),
                    mixup_cfg=mix_cfg,
                )
                logits = model(mixed_images)
                if mixed:
                    loss = soft_target_ce(logits, mixed_targets)
                else:
                    loss = focal_loss(logits, labels_batch) if use_focal else ce_loss(logits, labels_batch)

            if use_amp:
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_loss += loss.item() * labels_batch.size(0)
            total_samples += labels_batch.size(0)
            total_correct += (torch.argmax(logits, dim=1) == labels_batch).sum().item()

        avg_loss = total_loss / max(total_samples, 1)
        avg_acc = total_correct / max(total_samples, 1)
        return avg_loss, avg_acc

    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)

    def evaluate_and_update(
        loader: DataLoader,
        source: str,
        epoch: int,
    ) -> Dict[str, float]:
        nonlocal best_f1, best_state, best_metrics, best_epoch, best_source, no_improve
        metrics, labels_out, preds_out = evaluate(model, loader, device, labels)
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_state = model.state_dict()
            best_metrics = metrics
            best_epoch = epoch
            best_source = source
            no_improve = 0
            save_confusion(
                labels_out,
                preds_out,
                labels,
                run_dir / f"{source}_confusion.png",
            )
        else:
            no_improve += 1
        return metrics

    if phase_b_only:
        phase_a_ckpt = phase_b.get("phase_a_checkpoint")
        if not phase_a_ckpt:
            phase_a_ckpt = models_dir / "tier1_phase_a_latest.pt"
        phase_a_ckpt = Path(phase_a_ckpt)
        if not phase_a_ckpt.exists():
            raise FileNotFoundError(
                f"Phase A checkpoint not found: {phase_a_ckpt}"
            )
        state = torch.load(phase_a_ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"Phase A skipped. Loaded: {phase_a_ckpt}")
    else:
        print("Phase A: mixed dataset")
        optimizer = make_optimizer(float(phase_a.get("lr", 1e-3)))
        scheduler = make_scheduler(optimizer, int(phase_a.get("epochs", 10)))

        for epoch in range(1, phase_a_epochs + 1):
            train_loss, train_acc = run_epoch(train_loader, optimizer, mix_cfg_a)
            if scheduler:
                scheduler.step()

            val_metrics, _, _ = evaluate(model, val_loader, device, labels)
            source_metrics = val_metrics
            domain_metrics = None
            if use_domain_val:
                domain_metrics, _, _ = evaluate(
                    model, domain_val_loader, device, labels
                )
                source_metrics = domain_metrics

            if source_metrics["macro_f1"] > best_f1:
                best_state = model.state_dict()
                best_f1 = source_metrics["macro_f1"]
                best_metrics = source_metrics
                best_epoch = epoch
                best_source = (
                    "domain_val" if source_metrics is not val_metrics else "val"
                )
                no_improve = 0
            else:
                no_improve += 1

            history.append(
                {
                    "phase": 1,
                    "epoch": epoch,
                    "global_epoch": epoch,
                    "train_loss": float(train_loss),
                    "train_acc": float(train_acc),
                    "val_top1": float(val_metrics["top1"]),
                    "val_top3": float(val_metrics["top3"]),
                    "val_f1": float(val_metrics["macro_f1"]),
                }
            )
            msg = (
                f"Epoch {epoch}/{phase_a_epochs} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_top1={val_metrics['top1']:.4f} "
                f"val_top3={val_metrics['top3']:.4f} "
                f"val_f1={val_metrics['macro_f1']:.4f}"
            )
            if domain_metrics is not None:
                msg += (
                    f"\ndomain_top1={domain_metrics['top1']:.4f} "
                    f"domain_top3={domain_metrics['top3']:.4f} "
                    f"domain_f1={domain_metrics['macro_f1']:.4f}"
                )
            print(msg)
            log_wandb(
                wb_run,
                {
                    "phase": 1,
                    "epoch": epoch,
                    "global_epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_top1": val_metrics["top1"],
                    "val_top3": val_metrics["top3"],
                    "val_f1": val_metrics["macro_f1"],
                    "lr": optimizer.param_groups[0]["lr"],
                },
            )
            if domain_metrics is not None:
                log_wandb(
                    wb_run,
                    {
                        "domain_top1": domain_metrics["top1"],
                        "domain_top3": domain_metrics["top3"],
                        "domain_f1": domain_metrics["macro_f1"],
                    },
                )

            if early_patience > 0 and no_improve >= early_patience:
                print("Early stopping in Phase A")
                break

        phase_a_path = save_phase_a_latest(model, artifact_dir, models_dir)
        print(f"Saved Phase A latest -> {phase_a_path}")

    print("Phase B: domain fine-tune")
    phase_b_lr = float(phase_b.get("lr", 2e-4))
    phase_b_sched = phase_b.get("scheduler", {})
    freeze_epochs = int(phase_b.get("freeze_epochs", 0))
    head_lr = float(phase_b.get("head_lr", phase_b_lr))
    backbone_lr = float(phase_b.get("backbone_lr", phase_b_lr * 0.25))
    phase_b_total = phase_b_epochs
    phase_b_done = 0
    no_improve = 0

    backbone_params, head_params = split_backbone_head_params(model)

    if freeze_epochs > 0 and head_params:
        print(
            f"Phase B1: head-only fine-tune for {freeze_epochs} epoch(s) "
            f"(head_lr={head_lr})"
        )
        set_requires_grad(backbone_params, False)
        set_requires_grad(head_params, True)
        optimizer = make_optimizer_split(
            backbone_params=[],
            head_params=head_params,
            backbone_lr=backbone_lr,
            head_lr=head_lr,
        )
        scheduler = make_scheduler(
            optimizer,
            min(freeze_epochs, phase_b_total),
            override=phase_b_sched,
        )

        for epoch in range(1, min(freeze_epochs, phase_b_total) + 1):
            train_loss, train_acc = run_epoch(
                domain_train_loader, optimizer, mix_cfg_b
            )
            if scheduler:
                scheduler.step()
            source_metrics, _, _ = evaluate(
                model,
                domain_val_loader if use_domain_val else val_loader,
                device,
                labels,
            )
            phase_b_done += 1
            history.append(
                {
                    "phase": 2,
                    "epoch": phase_b_done,
                    "global_epoch": phase_a_epochs + phase_b_done,
                    "train_loss": float(train_loss),
                    "train_acc": float(train_acc),
                    "val_top1": float(source_metrics["top1"]),
                    "val_top3": float(source_metrics["top3"]),
                    "val_f1": float(source_metrics["macro_f1"]),
                }
            )
            print(
                f"Epoch {phase_b_done}/{phase_b_total} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_top1={source_metrics['top1']:.4f} "
                f"val_top3={source_metrics['top3']:.4f} "
                f"val_f1={source_metrics['macro_f1']:.4f}"
            )
            log_wandb(
                wb_run,
                {
                    "phase": 2,
                    "epoch": phase_b_done,
                    "global_epoch": phase_a_epochs + phase_b_done,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_top1": source_metrics["top1"],
                    "val_top3": source_metrics["top3"],
                    "val_f1": source_metrics["macro_f1"],
                    "lr": optimizer.param_groups[0]["lr"],
                },
            )

            if source_metrics["macro_f1"] > best_f1:
                best_state = model.state_dict()
                best_f1 = source_metrics["macro_f1"]
                best_metrics = source_metrics
                best_epoch = phase_b_done
                best_source = "domain_val"
                no_improve = 0
            else:
                no_improve += 1
            if early_patience > 0 and no_improve >= early_patience:
                print("Early stopping in Phase B1")
                break

    if phase_b_done < phase_b_total:
        remaining = phase_b_total - phase_b_done
        print(
            f"Phase B2: full fine-tune for {remaining} epoch(s) "
            f"(backbone_lr={backbone_lr}, head_lr={head_lr})"
        )
        set_requires_grad(backbone_params, True)
        set_requires_grad(head_params, True)
        optimizer = make_optimizer_split(
            backbone_params=backbone_params,
            head_params=head_params,
            backbone_lr=backbone_lr,
            head_lr=head_lr,
        )
        scheduler = make_scheduler(optimizer, remaining, override=phase_b_sched)
        no_improve = 0

        for epoch in range(1, remaining + 1):
            train_loss, train_acc = run_epoch(
                domain_train_loader, optimizer, mix_cfg_b
            )
            if scheduler:
                scheduler.step()

            source_metrics, _, _ = evaluate(
                model,
                domain_val_loader if use_domain_val else val_loader,
                device,
                labels,
            )
            phase_b_done += 1
            if source_metrics["macro_f1"] > best_f1:
                best_state = model.state_dict()
                best_f1 = source_metrics["macro_f1"]
                best_metrics = source_metrics
                best_epoch = phase_b_done
                best_source = "domain_val"
                no_improve = 0
            else:
                no_improve += 1

            history.append(
                {
                    "phase": 2,
                    "epoch": phase_b_done,
                    "global_epoch": phase_a_epochs + phase_b_done,
                    "train_loss": float(train_loss),
                    "train_acc": float(train_acc),
                    "val_top1": float(source_metrics["top1"]),
                    "val_top3": float(source_metrics["top3"]),
                    "val_f1": float(source_metrics["macro_f1"]),
                }
            )
            print(
                f"Epoch {phase_b_done}/{phase_b_total} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_top1={source_metrics['top1']:.4f} "
                f"val_top3={source_metrics['top3']:.4f} "
                f"val_f1={source_metrics['macro_f1']:.4f}"
            )
            log_wandb(
                wb_run,
                {
                    "phase": 2,
                    "epoch": phase_b_done,
                    "global_epoch": phase_a_epochs + phase_b_done,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_top1": source_metrics["top1"],
                    "val_top3": source_metrics["top3"],
                    "val_f1": source_metrics["macro_f1"],
                    "lr": optimizer.param_groups[0]["lr"],
                },
            )

            if early_patience > 0 and no_improve >= early_patience:
                print("Early stopping in Phase B2")
                break

    if best_state is None:
        best_state = model.state_dict()

    best_path = artifact_dir / "best.pt"
    torch.save(best_state, best_path)

    model.load_state_dict(best_state)
    val_metrics, val_labels, val_preds = evaluate(
        model, val_loader, device, labels
    )
    test_metrics, test_labels, test_preds = evaluate(
        model, test_loader, device, labels
    )
    domain_metrics = None
    domain_labels: List[int] = []
    domain_preds: List[int] = []
    if use_domain_val:
        domain_metrics, domain_labels, domain_preds = evaluate(
            model, domain_val_loader, device, labels
        )

    save_confusion(val_labels, val_preds, labels, run_dir / "val_confusion.png")
    save_confusion(test_labels, test_preds, labels, run_dir / "test_confusion.png")
    if domain_metrics is not None:
        save_confusion(
            domain_labels,
            domain_preds,
            labels,
            run_dir / "domain_val_confusion.png",
        )

    val_report = classification_report(
        val_labels,
        val_preds,
        labels=list(range(len(labels))),
        target_names=labels,
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    test_report = classification_report(
        test_labels,
        test_preds,
        labels=list(range(len(labels))),
        target_names=labels,
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    save_json(run_dir / "val_report.json", val_report)
    save_json(run_dir / "test_report.json", test_report)
    if domain_metrics is not None:
        domain_report = classification_report(
            domain_labels,
            domain_preds,
            labels=list(range(len(labels))),
            target_names=labels,
            digits=4,
            zero_division=0,
            output_dict=True,
        )
        save_json(run_dir / "domain_val_report.json", domain_report)

    save_json(run_dir / "val_metrics.json", val_metrics)
    save_json(run_dir / "test_metrics.json", test_metrics)
    if domain_metrics is not None:
        save_json(run_dir / "domain_val_metrics.json", domain_metrics)

    save_history(history, run_dir)
    for name in (
        "metrics.csv",
        "metrics.json",
        "curves.png",
        "val_confusion.png",
        "test_confusion.png",
        "domain_val_confusion.png",
        "val_report.json",
        "test_report.json",
        "domain_val_report.json",
        "val_metrics.json",
        "test_metrics.json",
        "domain_val_metrics.json",
    ):
        src = run_dir / name
        if src.exists():
            (artifact_dir / name).write_bytes(src.read_bytes())

    opset = int(cfg.get("onnx", {}).get("opset", 17))
    infer_cfg = cfg.get("infer", {}) or {}
    onnx_path = export_onnx_model(
        model=model,
        artifact_dir=artifact_dir,
        labels=labels,
        img_size=img_size,
        mean=mean,
        std=std,
        opset=opset,
        backbone=backbone,
        infer_cfg=infer_cfg,
    )
    update_models_dir(artifact_dir, models_dir)

    phase_summary = summarize_phases(history)
    summary = {
        "run_name": run_name,
        "best_epoch": best_epoch,
        "best_source": best_source,
        "best_metrics": best_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "domain_val_metrics": domain_metrics,
        "phase_summary": phase_summary,
        "config": cfg,
        "git_commit": get_git_commit(Path.cwd()),
    }
    save_json(artifact_dir / "run_summary.json", summary)
    if wb_run is not None:
        wb_run.summary.update(
            {
                "best_epoch": best_epoch,
                "best_source": best_source,
                "best_f1": best_f1,
                "val_top1": val_metrics["top1"],
                "val_top3": val_metrics["top3"],
                "val_f1": val_metrics["macro_f1"],
                "test_top1": test_metrics["top1"],
                "test_top3": test_metrics["top3"],
                "test_f1": test_metrics["macro_f1"],
            }
        )
        wb_run.finish()

    last_run_path = output_root / "last_run.txt"
    last_run_path.write_text(str(artifact_dir), encoding="utf-8")

    print(f"Saved best checkpoint -> {best_path}")
    print(f"Exported ONNX -> {onnx_path}")
    print(f"Updated models -> {models_dir}")
    print(
        f"Test top1={test_metrics['top1']:.4f} "
        f"top3={test_metrics['top3']:.4f} "
        f"macro_f1={test_metrics['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()

