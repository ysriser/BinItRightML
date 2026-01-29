from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader

from .transforms import build_transforms


def _make_weighted_sampler(dataset: ImageFolder) -> WeightedRandomSampler:
    counts = Counter(dataset.targets)
    class_weights = {cls: 1.0 / count for cls, count in counts.items()}
    weights = [class_weights[label] for label in dataset.targets]
    return WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True)


def compute_class_weights(dataset: ImageFolder) -> torch.Tensor:
    counts = Counter(dataset.targets)
    total = sum(counts.values())
    num_classes = len(dataset.classes)
    weights = torch.zeros(num_classes, dtype=torch.float)
    for cls, count in counts.items():
        weights[cls] = total / (num_classes * float(count))
    return weights


class FolderDataset(Dataset):
    """Image folder dataset that allows empty class folders."""

    def __init__(self, root: Path, class_to_idx: Dict[str, int], transform=None) -> None:
        self.root = Path(root)
        self.class_to_idx = class_to_idx
        self.classes = [c for c, _ in sorted(class_to_idx.items(), key=lambda x: x[1])]
        self.transform = transform
        self.samples = []
        self.targets = []

        for class_name, idx in class_to_idx.items():
            class_dir = self.root / class_name
            if not class_dir.exists():
                continue
            for path in class_dir.rglob("*"):
                if path.is_file() and path.suffix.lower() in IMG_EXTENSIONS:
                    self.samples.append((path, idx))
                    self.targets.append(idx)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        image = default_loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def create_dataloaders(
    data_dir: Path,
    backbone: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    persistent_workers: bool,
    prefetch_factor: int,
    use_weighted_sampler: bool,
    device: torch.device,
) -> Tuple[Dict[str, DataLoader], torch.Tensor, List[str]]:
    train_dir = Path(data_dir) / "train"
    val_dir = Path(data_dir) / "val"
    test_dir = Path(data_dir) / "test"

    for split_dir in (train_dir, val_dir, test_dir):
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")

    train_tfm, eval_tfm = build_transforms(backbone, image_size)
    train_ds = ImageFolder(train_dir, transform=train_tfm)

    # Keep class_to_idx from train and allow empty classes in val/test.
    class_to_idx = train_ds.class_to_idx
    val_ds = FolderDataset(val_dir, class_to_idx, transform=eval_tfm)
    test_ds = FolderDataset(test_dir, class_to_idx, transform=eval_tfm)

    sampler = _make_weighted_sampler(train_ds) if use_weighted_sampler else None
    class_weights = compute_class_weights(train_ds)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor

    loaders = {
        "train": DataLoader(train_ds, shuffle=sampler is None, sampler=sampler, **loader_kwargs),
        "val": DataLoader(val_ds, shuffle=False, **loader_kwargs),
        "test": DataLoader(test_ds, shuffle=False, **loader_kwargs),
    }
    return loaders, class_weights, train_ds.classes
