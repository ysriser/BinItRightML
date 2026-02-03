"""Robust fine-tune v2 training for Tier-1 (7-class) classifier."""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
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
    def __init__(self, items: List[Tuple[Path, int]], transform=None) -> None:
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        with Image.open(path) as opened:
            img = opened.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class ScaleDownPad:
    def __init__(
        self,
        img_size: int,
        scale_range: Tuple[float, float],
        prob: float,
        random_position: bool,
    ) -> None:
        self.img_size = img_size
        self.scale_range = scale_range
        self.prob = prob
        self.random_position = random_position

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.prob:
            return img
        scale = random.uniform(*self.scale_range)
        new_size = max(1, int(round(self.img_size * scale)))
        resized = img.resize((new_size, new_size), resample=Image.BICUBIC)
        canvas = Image.new("RGB", (self.img_size, self.img_size), (0, 0, 0))
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


def build_train_transform(
    cfg: dict,
    img_size: int,
    mean: Sequence[float],
    std: Sequence[float],
):
    aug = cfg.get("augment", {})
    scale_min = float(aug.get("rrc_scale_min", 0.15))
    scale_max = float(aug.get("rrc_scale_max", 1.0))
    ratio = aug.get("rrc_ratio", [0.75, 1.33])
    color_jitter = aug.get("color_jitter", [0.3, 0.3, 0.3, 0.1])
    hflip_p = float(aug.get("hflip_p", 0.5))

    scale_cfg = aug.get("scale_down_pad", {}) or {}
    scale_enabled = bool(scale_cfg.get("enabled", True))

    affine_cfg = aug.get("affine", {}) or {}
    perspective_cfg = aug.get("perspective", {}) or {}
    blur_cfg = aug.get("blur", {}) or {}

    transforms_list: List[transforms.Transform] = [
        transforms.RandomResizedCrop(
            img_size,
            scale=(scale_min, scale_max),
            ratio=tuple(ratio),
            interpolation=InterpolationMode.BICUBIC,
        ),
    ]

    if scale_enabled:
        transforms_list.append(
            ScaleDownPad(
                img_size=img_size,
                scale_range=tuple(scale_cfg.get("scale_range", [0.3, 0.7])),
                prob=float(scale_cfg.get("prob", 0.5)),
                random_position=bool(scale_cfg.get("random_position", True)),
            )
        )

    transforms_list.extend(
        [
            transforms.RandomHorizontalFlip(p=hflip_p),
            transforms.ColorJitter(*color_jitter),
        ]
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

    erasing_p = float(aug.get("random_erasing_p", 0.0))
    if erasing_p > 0:
        transforms_list.append(transforms.RandomErasing(p=erasing_p))

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

    train_transform = build_train_transform(cfg, img_size, mean, std)
    eval_transform = build_eval_transform(img_size, mean, std)

    data_cfg = cfg.get("data", {})
    loader_kwargs = {
        "batch_size": int(data_cfg.get("batch_size", 32)),
        "num_workers": int(data_cfg.get("num_workers", 4)),
        "pin_memory": bool(data_cfg.get("pin_memory", True)),
        "persistent_workers": bool(data_cfg.get("persistent_workers", True)),
        "prefetch_factor": int(data_cfg.get("prefetch_factor", 2)),
        "worker_init_fn": seed_worker,
    }

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

    run_name = datetime.now().strftime("robust_v2_%Y%m%d_%H%M%S")
    run_dir = output_root / run_name
    artifact_dir = artifact_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "config_snapshot.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    phase_a = cfg.get("phase_a", {})
    phase_b = cfg.get("phase_b", {})

    train_ds = CsvDataset(
        train_csv,
        label_to_idx,
        transform=train_transform,
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

    train_loader = DataLoader(train_ds, sampler=sampler, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

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
    domain_train_ds = ListDataset(domain_split.train, transform=train_transform)
    domain_val_ds = ListDataset(domain_split.val, transform=eval_transform)
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
        **loader_kwargs,
    )
    domain_val_loader = DataLoader(domain_val_ds, shuffle=False, **loader_kwargs)

    loss_cfg = cfg.get("loss", {})
    label_smoothing = float(loss_cfg.get("label_smoothing", 0.05))
    ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    mix_cfg = cfg.get("mixup_cutmix", {}) or {}
    scale_cfg = cfg.get("augment", {}).get("scale_down_pad", {}) or {}
    print(
        "Augmentations -> "
        f"scale_down_pad={scale_cfg.get('enabled', True)} "
        f"mixup_p={mix_cfg.get('mixup_p', 0.0)} "
        f"cutmix_p={mix_cfg.get('cutmix_p', 0.0)}"
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

    sched_cfg = cfg.get("scheduler", {})
    scheduler_enabled = bool(sched_cfg.get("enabled", True))
    scheduler_name = sched_cfg.get("name", "cosine")
    min_lr = float(sched_cfg.get("min_lr", 1e-6))

    def make_scheduler(optimizer: torch.optim.Optimizer, epochs: int):
        if not scheduler_enabled:
            return None
        if scheduler_name == "cosine":
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

    def run_epoch(
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
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
                    loss = ce_loss(logits, labels_batch)

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

    print("Phase A: mixed dataset")
    optimizer = make_optimizer(float(phase_a.get("lr", 1e-3)))
    scheduler = make_scheduler(optimizer, int(phase_a.get("epochs", 10)))

    for epoch in range(1, int(phase_a.get("epochs", 10)) + 1):
        train_loss, train_acc = run_epoch(train_loader, optimizer)
        if scheduler:
            scheduler.step()

        val_metrics, _, _ = evaluate(model, val_loader, device, labels)
        source_metrics = val_metrics
        if use_domain_val:
            source_metrics, _, _ = evaluate(model, domain_val_loader, device, labels)

        if source_metrics["macro_f1"] > best_f1:
            best_state = model.state_dict()
            best_f1 = source_metrics["macro_f1"]
            best_metrics = source_metrics
            best_epoch = epoch
            best_source = "domain_val" if source_metrics is not val_metrics else "val"
            no_improve = 0
        else:
            no_improve += 1

        print(
            f"Epoch {epoch}/{phase_a.get('epochs', 10)} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_f1={val_metrics['macro_f1']:.4f}"
        )

        if early_patience > 0 and no_improve >= early_patience:
            print("Early stopping in Phase A")
            break

    print("Phase B: domain fine-tune")
    optimizer = make_optimizer(float(phase_b.get("lr", 2e-4)))
    scheduler = make_scheduler(optimizer, int(phase_b.get("epochs", 5)))
    no_improve = 0

    for epoch in range(1, int(phase_b.get("epochs", 5)) + 1):
        train_loss, train_acc = run_epoch(domain_train_loader, optimizer)
        if scheduler:
            scheduler.step()

        source_metrics, _, _ = evaluate(
            model,
            domain_val_loader if use_domain_val else val_loader,
            device,
            labels,
        )
        if source_metrics["macro_f1"] > best_f1:
            best_state = model.state_dict()
            best_f1 = source_metrics["macro_f1"]
            best_metrics = source_metrics
            best_epoch = epoch
            best_source = "domain_val"
            no_improve = 0
        else:
            no_improve += 1

        print(
            f"Epoch {epoch}/{phase_b.get('epochs', 5)} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"domain_f1={source_metrics['macro_f1']:.4f}"
        )

        if early_patience > 0 and no_improve >= early_patience:
            print("Early stopping in Phase B")
            break

    if best_state is None:
        best_state = model.state_dict()

    best_path = artifact_dir / "best.pt"
    torch.save(best_state, best_path)

    model.load_state_dict(best_state)
    test_metrics, test_labels, test_preds = evaluate(model, test_loader, device, labels)
    save_confusion(test_labels, test_preds, labels, run_dir / "test_confusion.png")

    report = classification_report(
        test_labels,
        test_preds,
        labels=list(range(len(labels))),
        target_names=labels,
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    save_json(run_dir / "test_report.json", report)

    infer_cfg = {
        "backbone": backbone,
        "img_size": img_size,
        "mean": mean,
        "std": std,
        "topk": cfg.get("infer", {}).get("topk", 3),
        "conf_threshold": cfg.get("infer", {}).get("conf_threshold", 0.75),
        "margin_threshold": cfg.get("infer", {}).get("margin_threshold", 0.15),
    }
    save_json(artifact_dir / "infer_config.json", infer_cfg)
    save_json(artifact_dir / "label_map.json", {"labels": labels})

    summary = {
        "run_name": run_name,
        "best_epoch": best_epoch,
        "best_source": best_source,
        "best_metrics": best_metrics,
        "test_metrics": test_metrics,
        "config": cfg,
        "git_commit": get_git_commit(Path.cwd()),
    }
    save_json(artifact_dir / "run_summary.json", summary)

    last_run_path = output_root / "last_run.txt"
    last_run_path.write_text(str(artifact_dir), encoding="utf-8")

    print(f"Saved best checkpoint -> {best_path}")
    print(
        f"Test top1={test_metrics['top1']:.4f} "
        f"top3={test_metrics['top3']:.4f} "
        f"macro_f1={test_metrics['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
