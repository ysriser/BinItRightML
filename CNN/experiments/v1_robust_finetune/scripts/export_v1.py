"""Export robust fine-tune v1 checkpoint to ONNX."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List

import numpy as np
import onnxruntime as ort
import timm
import torch
import yaml
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision import transforms

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


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("train_v1.yaml must be a YAML mapping")
    return data


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def list_images(data_dir: Path, limit: int) -> List[Path]:
    items: List[Path] = []
    for file in sorted(data_dir.rglob("*")):
        if file.is_file() and file.suffix.lower() in IMAGE_EXTS:
            items.append(file)
    return items[:limit]


def list_images_from_csv(csv_path: Path, limit: int) -> List[Path]:
    items: List[Path] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_path = row.get("filepath")
            if not raw_path:
                continue
            path = Path(raw_path)
            if not path.is_absolute():
                path = csv_path.parent / path
            if path.exists():
                items.append(path)
            if len(items) >= limit:
                break
    return items


def build_transform(img_size: int, mean: List[float], std: List[float]):
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export robust fine-tune v1")
    p.add_argument(
        "--config",
        type=Path,
        default=Path("CNN/experiments/v1_robust_finetune/configs/train_v1.yaml"),
    )
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--artifact-dir", type=Path, default=None)
    p.add_argument("--sanity-images", type=Path, default=None)
    p.add_argument("--prefer-cuda", action="store_true", default=False)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    paths = cfg.get("paths", {})
    artifact_root = Path(
        paths.get(
            "artifact_dir",
            "CNN/experiments/v1_robust_finetune/artifacts",
        )
    )
    output_root = Path(
        paths.get(
            "output_dir",
            "CNN/experiments/v1_robust_finetune/outputs",
        )
    )

    if args.artifact_dir:
        artifact_dir = args.artifact_dir
    else:
        last_run = output_root / "last_run.txt"
        if not last_run.exists():
            raise FileNotFoundError("last_run.txt not found; provide --artifact-dir")
        artifact_dir = Path(last_run.read_text(encoding="utf-8").strip())

    checkpoint = args.checkpoint or (artifact_dir / "best.pt")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

    labels = cfg.get("labels", [])
    if not labels:
        raise ValueError("labels is empty in train_v1.yaml")

    model_cfg = cfg.get("model", {})
    backbone = model_cfg.get("backbone", "efficientnet_b0")
    img_size = int(model_cfg.get("img_size", 224))

    model = timm.create_model(backbone, pretrained=False, num_classes=len(labels))
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    data_cfg_model = timm.data.resolve_data_config({}, model=model)
    mean = data_cfg_model["mean"]
    std = data_cfg_model["std"]

    dummy = torch.randn(1, 3, img_size, img_size)
    onnx_path = artifact_dir / "model.onnx"

    opset = int(cfg.get("onnx", {}).get("opset", 17))
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
    )

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

    sanity_limit = int(cfg.get("onnx", {}).get("sanity_images", 5))
    sanity_path = args.sanity_images
    samples: List[Path] = []

    if sanity_path is not None:
        if sanity_path.is_dir():
            samples = list_images(sanity_path, sanity_limit)
        elif sanity_path.is_file():
            samples = list_images_from_csv(sanity_path, sanity_limit)
    else:
        g3_dir = Path(paths.get("g3_data_dir", "CNN/data/G3_SGData"))
        if g3_dir.exists():
            samples = list_images(g3_dir, sanity_limit)
        else:
            train_csv = Path(paths.get("train_csv", ""))
            if train_csv.exists():
                samples = list_images_from_csv(train_csv, sanity_limit)
    transform = build_transform(img_size, mean, std)

    providers = ["CPUExecutionProvider"]
    if args.prefer_cuda and "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    session = ort.InferenceSession(str(onnx_path), providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    if not samples:
        print("No sanity images found; skipping ONNX sanity check.")
    else:
        for path in samples:
            with Image.open(path) as opened:
                img = opened.convert("RGB")
            tensor = transform(img).unsqueeze(0).numpy().astype(np.float32)
            outputs = session.run([output_name], {input_name: tensor})
            logits = outputs[0]
            if logits.shape[-1] != len(labels):
                raise ValueError("ONNX output size mismatch")

    print(f"Exported ONNX -> {onnx_path}")
    print(f"Sanity checked {len(samples)} images")


if __name__ == "__main__":
    main()
