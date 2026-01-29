import argparse
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.models.classifier import build_classifier  # noqa: E402
from src.utils import ensure_dir, load_checkpoint, load_yaml, save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export TACO super-28 model")
    p.add_argument("--config", type=Path, default=Path("ml/classification_28cats/configs/taco_super28.yaml"))
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--onnx", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device", "cpu") == "cuda" else "cpu")

    ckpt = load_checkpoint(args.checkpoint, device=device)
    class_names = ckpt["class_names"]
    backbone = ckpt.get("backbone", cfg.get("backbone", "efficientnet_b0"))
    image_size = ckpt.get("image_size", cfg.get("image_size", 224))

    model, _ = build_classifier(len(class_names), backbone=backbone, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    run_name = Path(args.checkpoint).parent.name
    artifact_dir = Path(cfg.get("artifact_dir", "ml/artifacts/classification_28cats")) / run_name
    ensure_dir(artifact_dir)

    model_path = artifact_dir / "model.ts"
    dummy = torch.randn(1, 3, image_size, image_size, device=device)
    traced = torch.jit.trace(model, dummy)
    traced.save(model_path)

    labels_path = artifact_dir / "model.labels.json"
    save_json(
        {
            "class_names": class_names,
            "backbone": backbone,
            "image_size": image_size,
            "model_version": ckpt.get("timestamp", "unknown"),
        },
        labels_path,
    )

    if args.onnx:
        onnx_path = artifact_dir / "model.onnx"
        torch.onnx.export(
            model,
            dummy,
            onnx_path,
            input_names=["image"],
            output_names=["logits"],
            opset_version=12,
        )

    # Also write a "latest" copy for serving convenience
    latest_dir = Path(cfg.get("artifact_dir", "ml/artifacts/classification_28cats")) / "latest"
    ensure_dir(latest_dir)
    shutil.copy2(model_path, latest_dir / "model.ts")
    shutil.copy2(labels_path, latest_dir / "model.labels.json")

    print(f"Saved TorchScript to {model_path}")
    if args.onnx:
        print(f"Saved ONNX to {onnx_path}")
    print(f"Latest copy -> {latest_dir}")


if __name__ == "__main__":
    main()
