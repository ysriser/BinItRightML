from pathlib import Path

import torch
from PIL import Image
from torchvision.datasets import ImageFolder

from src.data.splits import FolderDataset, compute_class_weights, create_dataloaders
from src.data.transforms import build_transforms


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (32, 32), color=(128, 128, 128))
    img.save(path)


def _make_split(root: Path, split: str, class_names: list[str]) -> None:
    for class_name in class_names:
        _write_image(root / split / class_name / "img1.jpg")


def test_build_transforms_outputs_tensor():
    train_tfm, eval_tfm = build_transforms("efficientnet_b0", image_size=224)
    img = Image.new("RGB", (256, 256), color=(10, 20, 30))
    train_out = train_tfm(img)
    eval_out = eval_tfm(img)
    assert train_out.shape == (3, 224, 224)
    assert eval_out.shape == (3, 224, 224)


def test_compute_class_weights_balanced(tmp_path: Path):
    _make_split(tmp_path, "train", ["a", "b"])
    ds = ImageFolder(tmp_path / "train")
    weights = compute_class_weights(ds)
    assert torch.allclose(weights, torch.tensor([1.0, 1.0]))


def test_folder_dataset_allows_missing_classes(tmp_path: Path):
    class_to_idx = {"a": 0, "b": 1}
    _write_image(tmp_path / "val" / "a" / "img1.jpg")
    ds = FolderDataset(tmp_path / "val", class_to_idx)
    assert len(ds) == 1
    assert ds.classes == ["a", "b"]


def test_create_dataloaders_basic(tmp_path: Path):
    for split in ("train", "val", "test"):
        _make_split(tmp_path, split, ["a", "b"])
    loaders, class_weights, class_names = create_dataloaders(
        data_dir=tmp_path,
        backbone="efficientnet_b0",
        image_size=64,
        batch_size=2,
        num_workers=0,
        persistent_workers=False,
        prefetch_factor=2,
        use_weighted_sampler=False,
        device=torch.device("cpu"),
    )
    assert set(loaders.keys()) == {"train", "val", "test"}
    assert class_names == ["a", "b"]
    assert class_weights.shape[0] == 2
