import argparse
import io
from pathlib import Path
from typing import Any, Dict, List

import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from src.data.transforms import build_transforms
from src.rules import apply_rules
from src.router import Router
from src.utils import load_json, load_yaml


def load_artifacts(artifact: Path, labels: Path, device: torch.device):
    meta = load_json(labels)
    backbone = meta.get("backbone", "efficientnet_b0")
    image_size = meta.get("image_size", 224)
    _, eval_tfm = build_transforms(backbone, image_size)
    model = torch.jit.load(artifact, map_location=device)
    model.to(device)
    model.eval()
    return model, meta, eval_tfm


def create_app(artifact: Path, labels: Path, config: Path, device: torch.device) -> FastAPI:
    cfg = load_yaml(config)
    model, meta, eval_tfm = load_artifacts(artifact, labels, device)
    class_names: List[str] = meta["class_names"]
    version = meta.get("model_version", "unknown")
    threshold = float(cfg.get("confidence_threshold", 0.6))
    high_risk = list(cfg.get("high_risk_categories", []))
    router = Router()

    app = FastAPI(title="Bin-It-Right TACO Super-28", version=version)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {"status": "ok"}

    @app.post("/api/v1/scan")
    async def scan(image: UploadFile = File(...)) -> Dict[str, Any]:
        if not image.filename:
            raise HTTPException(status_code=400, detail="No image uploaded.")
        try:
            contents = await image.read()
            pil = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception as exc:  # pylint: disable=broad-except
            raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

        tensor = eval_tfm(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0]

        topk = torch.topk(probs, k=min(3, len(class_names)))
        top1_idx = topk.indices[0].item()
        top1_conf = float(topk.values[0].item())
        category = class_names[top1_idx]
        top3 = [
            {"category": class_names[idx.item()], "confidence": float(val.item())}
            for val, idx in zip(topk.values, topk.indices)
        ]

        rule_payload = apply_rules(category, top1_conf, threshold, high_risk)
        base_result = {
            "category": category,
            "recyclable": rule_payload["recyclable"],
            "confidence": top1_conf,
            "instructions": rule_payload["instructions"],
            "top3": top3,
            "needs_confirmation": rule_payload["needs_confirmation"],
            "followup_questions": rule_payload["followup_questions"],
        }

        result = router.route(category, base_result)
        return result

    return app


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serve TACO super-28 classifier")
    p.add_argument("--config", type=Path, default=Path("ml/classification_28cats/configs/taco_super28.yaml"))
    p.add_argument("--artifact", type=Path, default=Path("ml/artifacts/classification_28cats/latest/model.ts"))
    p.add_argument("--labels", type=Path, default=Path("ml/artifacts/classification_28cats/latest/model.labels.json"))
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    app = create_app(args.artifact, args.labels, args.config, device)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
