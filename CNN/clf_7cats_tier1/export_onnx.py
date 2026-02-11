"""
Export Tier-1 model to ONNX and optionally sanity-check with ONNX Runtime.
Run:
  python CNN/clf_7cats_tier1/export_onnx.py --check-image <path_to_image>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import timm
import torch
from PIL import Image
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]


def load_torch_weights(path: Path, map_location: str = "cpu"):
    """Load state dict with safe defaults and backward compatibility."""
    try:
        return torch.load(  # nosec B614 - loading trusted internal training artifact
            path,
            map_location=map_location,
            weights_only=True,
        )
    except TypeError:
        # Older torch versions do not support weights_only.
        return torch.load(  # nosec B614 - loading trusted internal training artifact
            path,
            map_location=map_location,
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export Tier-1 ONNX + optional ORT check")
    p.add_argument("--model-dir", type=Path, default=ROOT / "models")
    p.add_argument("--output", type=Path, default=None, help="Default: <model-dir>/tier1.onnx")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--dynamic-batch", action="store_true", help="Enable dynamic batch axis")
    p.add_argument("--check-image", type=Path, default=None, help="Image path for PT vs ORT check")
    p.add_argument("--warmup", type=int, default=5, help="Warmup runs for ORT timing")
    p.add_argument("--runs", type=int, default=20, help="Measured runs for ORT timing")
    return p.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_transform(img_size: int, mean: List[float], std: List[float]) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.15)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def load_model(model_dir: Path) -> Tuple[torch.nn.Module, Dict[str, Any], List[str]]:
    label_map = load_json(model_dir / "label_map.json")
    infer_cfg = load_json(model_dir / "infer_config.json")
    backbone = infer_cfg.get("backbone", "efficientnet_b0")
    labels = label_map["labels"]

    model = timm.create_model(backbone, pretrained=False, num_classes=len(labels))
    state = load_torch_weights(model_dir / "tier1_best.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, infer_cfg, labels


def softmax_np(x):
    import numpy as np

    x = x - x.max(axis=1, keepdims=True)
    exp = np.exp(x)
    return exp / exp.sum(axis=1, keepdims=True)


def topk(probs, labels: List[str], k: int) -> List[Tuple[str, float]]:
    import numpy as np

    idxs = np.argsort(-probs)[:k]
    return [(labels[i], float(probs[i])) for i in idxs]


def export_onnx(model: torch.nn.Module, img_size: int, out_path: Path, opset: int, dynamic_batch: bool) -> None:
    dummy = torch.randn(1, 3, img_size, img_size)
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {"input": {0: "batch"}, "logits": {0: "batch"}}

    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
    )


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir
    out_path = args.output or (model_dir / "tier1.onnx")

    model, infer_cfg, labels = load_model(model_dir)
    img_size = int(infer_cfg.get("img_size", 224))

    export_onnx(model, img_size, out_path, args.opset, args.dynamic_batch)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"ONNX saved: {out_path} ({size_mb:.2f} MB)")

    if not args.check_image:
        return

    try:
        import onnxruntime as ort
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"onnxruntime not available: {exc}")
        return

    transform = build_transform(img_size, infer_cfg["mean"], infer_cfg["std"])
    img = Image.open(args.check_image).convert("RGB")
    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        pt_logits = model(tensor)
        pt_probs = torch.softmax(pt_logits, dim=1).squeeze(0).cpu().numpy()

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    ort_logits = sess.run(None, {input_name: tensor.cpu().numpy()})[0]
    ort_probs = softmax_np(ort_logits).squeeze(0)

    pt_top3 = topk(pt_probs, labels, 3)
    ort_top3 = topk(ort_probs, labels, 3)

    print("PT top3:", pt_top3)
    print("ORT top3:", ort_top3)

    for _ in range(args.warmup):
        _ = sess.run(None, {input_name: tensor.cpu().numpy()})

    start = time.perf_counter()
    for _ in range(args.runs):
        _ = sess.run(None, {input_name: tensor.cpu().numpy()})
    elapsed_ms = (time.perf_counter() - start) * 1000 / max(1, args.runs)
    print(f"ORT avg latency: {elapsed_ms:.1f} ms (CPU, batch=1)")


if __name__ == "__main__":
    main()
