"""
FastAPI inference for Tier-1 model.
Run:
  uvicorn ml.serve_fastapi:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict, List

import timm
import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from torchvision import transforms

MODEL_DIR = Path("ml/models")

RULES = {
    "paper": ["Empty and dry before recycling.", "Remove food residue."],
    "plastic": ["Rinse and empty containers.", "Remove caps if possible."],
    "metal": ["Rinse cans and remove residue."],
    "glass": ["Rinse glass containers.", "Avoid breaking glass."],
    "e-waste": ["Do not bin. Bring to e-waste collection point."],
    "textile": ["Donate if usable; otherwise bag it for drop-off."],
    "other_uncertain": ["Use expert scan or check local guidelines."],
}


def run_expert_vlm(image_bytes: bytes, tier1_top3: List[Dict[str, Any]]) -> None:
    # Placeholder for expert model integration.
    return None


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_transform(img_size: int, mean: List[float], std: List[float]):
    return transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.15)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def load_model() -> Dict[str, Any]:
    label_map = load_json(MODEL_DIR / "label_map.json")
    infer_cfg = load_json(MODEL_DIR / "infer_config.json")
    backbone = infer_cfg.get("backbone", "efficientnet_b0")
    labels = label_map["labels"]

    model = timm.create_model(backbone, pretrained=False, num_classes=len(labels))
    state = torch.load(MODEL_DIR / "tier1_best.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    transform = build_transform(infer_cfg["img_size"], infer_cfg["mean"], infer_cfg["std"])
    return {"model": model, "labels": labels, "cfg": infer_cfg, "device": device, "transform": transform}


app = FastAPI()
STATE = load_model()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/scan")
async def scan(image: UploadFile = File(...)) -> Dict[str, Any]:
    image_bytes = await image.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = STATE["transform"](img).unsqueeze(0).to(STATE["device"])

    with torch.no_grad():
        logits = STATE["model"](tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    topk = int(STATE["cfg"].get("topk", 3))
    conf_threshold = float(STATE["cfg"].get("conf_threshold", 0.75))
    margin_threshold = float(STATE["cfg"].get("margin_threshold", 0.15))

    top_vals, top_idxs = torch.topk(probs, k=min(topk, probs.numel()))
    top3 = [
        {"label": STATE["labels"][idx], "p": float(val)}
        for idx, val in zip(top_idxs.tolist(), top_vals.tolist())
    ]

    top1_label = top3[0]["label"]
    top1_conf = top3[0]["p"]
    top2_conf = top3[1]["p"] if len(top3) > 1 else 0.0

    reasons: List[str] = []
    if top1_label == "other_uncertain":
        reasons.append("label_other_uncertain")
    if top1_conf < conf_threshold:
        reasons.append("low_confidence")
    if (top1_conf - top2_conf) < margin_threshold:
        reasons.append("small_margin")
    if top1_label in {"plastic", "glass"} and top1_conf < 0.85:
        reasons.append("strict_material_low_conf")

    escalate = len(reasons) > 0
    if escalate:
        run_expert_vlm(image_bytes, top3)

    instructions = RULES.get(top1_label, ["Follow local recycling guidelines."])

    return {
        "category": top1_label,
        "confidence": float(top1_conf),
        "top3": top3,
        "escalate": escalate,
        "instructions": instructions,
        "expert": None,
        "debug": {"reason": reasons, "model": STATE["cfg"].get("backbone", "unknown")},
    }
