"""V1 FastAPI server using ONNX + multi-crop rejection."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from CNN.shared import decision, multicrop, onnx_infer, preprocess

RULES = {
    "paper": ["Empty and dry before recycling.", "Remove food residue."],
    "plastic": ["Rinse and empty containers.", "Remove caps if possible."],
    "metal": ["Rinse cans and remove residue."],
    "glass": ["Rinse glass containers.", "Avoid breaking glass."],
    "e-waste": ["Do not bin. Bring to e-waste collection point."],
    "textile": ["Donate if usable; otherwise bag it for drop-off."],
    "other_uncertain": ["Use expert scan or check local guidelines."],
}


def run_expert_vlm(_: bytes, __: List[Dict[str, Any]]) -> None:
    # Placeholder for future expert model.
    return None


def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML mapping: {path}")
    return data


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def merge_paths(base: dict, override: dict) -> dict:
    for key in ("models", "data", "outputs"):
        if key in override:
            base.setdefault(key, {})
            base[key].update(override[key] or {})
    return base


def resolve_path(path_str: str, repo_root: Path) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else repo_root / path


CNN_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = CNN_DIR.parent
CONFIG_DIR = CNN_DIR / "experiments" / "v1_multicrop_reject" / "configs"

INFER_CFG_PATH = CONFIG_DIR / "infer_v1.yaml"
PATHS_CFG_PATH = CONFIG_DIR / "paths.yaml"

INFER_CFG = load_yaml(INFER_CFG_PATH)
INFER_CFG = merge_paths(INFER_CFG, load_yaml(PATHS_CFG_PATH))

MODEL_PATHS = INFER_CFG.get("models", {})
ONNX_PATH = resolve_path(
    str(MODEL_PATHS.get("onnx", "CNN/models/tier1.onnx")),
    REPO_ROOT,
)
LABEL_MAP_PATH = resolve_path(
    str(MODEL_PATHS.get("label_map", "CNN/models/label_map.json")),
    REPO_ROOT,
)
INFER_JSON_PATH = resolve_path(
    str(MODEL_PATHS.get("infer_config", "CNN/models/infer_config.json")), REPO_ROOT
)

LABEL_MAP = load_json(LABEL_MAP_PATH)
LABELS = preprocess.resolve_labels(LABEL_MAP)
preprocess.validate_labels(LABELS)

INFER_JSON = load_json(INFER_JSON_PATH)
IMG_SIZE = int(INFER_JSON.get("img_size", 224))
MEAN, STD = preprocess.resolve_mean_std(INFER_JSON)

OUTPUT_CFG = INFER_CFG.get("output", {}) or {}
TOPK = int(OUTPUT_CFG.get("topk", INFER_JSON.get("topk", 3)))

THRESHOLDS = INFER_CFG.get("thresholds", {}) or {}
THRESHOLDS["topk"] = TOPK

MULTICROP_CFG = INFER_CFG.get("multicrop", {}) or {}

ONNX_MODEL = onnx_infer.OnnxInfer(ONNX_PATH)


def infer_fn(batch: Any, _: str | None = None):
    return ONNX_MODEL.run(batch)


app = FastAPI()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/scan")
async def scan(image: UploadFile = File(...)) -> Dict[str, Any]:
    image_bytes = await image.read()
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = preprocess.ensure_rgb(img)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

    probs, mc_meta = multicrop.run_multicrop(
        img=img,
        infer_fn=infer_fn,
        img_size=IMG_SIZE,
        mean=MEAN,
        std=STD,
        cfg=MULTICROP_CFG,
    )

    result = decision.decide_from_probs(probs, LABELS, THRESHOLDS)
    if result["escalate"]:
        run_expert_vlm(image_bytes, result["top3"])

    category = result["final_label"]
    instructions = RULES.get(category, ["Follow local recycling guidelines."])

    return {
        "category": category,
        "confidence": float(result["top1_prob"]),
        "top3": result["top3"],
        "escalate": bool(result["escalate"]),
        "instructions": instructions,
        "expert": None,
        "debug": {
            "reasons": result["reasons"],
            "max_prob": float(result["metrics"]["max_prob"]),
            "margin": float(result["metrics"]["margin"]),
            "entropy": float(result["metrics"]["entropy"]),
            "multicrop_used": bool(mc_meta.get("multicrop_used")),
            "selected_crop": str(mc_meta.get("selected_crop")),
        },
    }
