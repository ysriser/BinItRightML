"""BinItRight Scan Service (v0.1 contract).

This FastAPI app implements the Android-facing endpoint defined in:
`CNN/docs/SCAN_SERVICE_SPEC_v0_1.md`

Goals for v0.1:
- Accept multipart upload: `image` (required), `tier1` (optional JSON string), `timestamp` (optional).
- Return a strict envelope: {status, request_id, data{decision, final, ...}}
- Tier-2 is NOT implemented yet. We always set decision.used_tier2=false.

Run (from repo root):
  uvicorn CNN.services.scan_service_v0_1:app --host 0.0.0.0 --port 8000

Quick curl:
  curl -X POST http://127.0.0.1:8000/api/v1/scan -F "image=@test.jpg"
"""

from __future__ import annotations

import io
import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from CNN.shared.scan_contract_v0_1 import (
    ScanResponse,
    Tier1Result,
    final_from_tier1,
    map_reason_codes,
)


SCHEMA_VERSION = "0.1"

TIER1_LABELS = {
    "paper",
    "plastic",
    "metal",
    "glass",
    "e-waste",
    "textile",
    "other_uncertain",
}


def _error(code: str, message: str, status_code: int) -> JSONResponse:
    payload: ScanResponse = {
        "status": "error",
        "request_id": str(uuid4()),
        "code": code,
        "message": message,
        "data": None,
    }
    return JSONResponse(status_code=status_code, content=payload)


def _load_tier1_thresholds() -> Dict[str, float]:
    """Load baseline thresholds from the canonical infer_config.json.

    These are echoed in `data.decision.thresholds` for observability. They do NOT
    need to match Android perfectly for v0.1, but keeping them aligned helps.
    """

    infer_path = Path("CNN/models/infer_config.json")
    try:
        infer_cfg = json.loads(infer_path.read_text(encoding="utf-8"))
    except Exception:
        # Safe fallback if the file is missing in a minimal server deployment.
        infer_cfg = {}

    return {
        "conf_threshold": float(infer_cfg.get("conf_threshold", 0.75)),
        "margin_threshold": float(infer_cfg.get("margin_threshold", 0.15)),
    }


def _parse_tier1_json(tier1_json: str) -> tuple[Optional[Tier1Result], list[str]]:
    """Parse Android-provided Tier-1 JSON (string field in multipart)."""

    reasons: list[str] = []
    try:
        raw = json.loads(tier1_json)
    except Exception as exc:  # noqa: BLE001 - user input parsing
        raise ValueError(f"Invalid tier1 JSON: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError("tier1 must be a JSON object")

    category = str(raw.get("category", "")).strip()
    if category not in TIER1_LABELS:
        reasons.append("unknown_tier1_label")
        category = "other_uncertain"

    try:
        confidence = float(raw.get("confidence", 0.0))
    except Exception:  # noqa: BLE001 - user input parsing
        confidence = 0.0
        reasons.append("invalid_tier1_confidence")

    escalate = bool(raw.get("escalate", False))

    top3_raw = raw.get("top3", [])
    top3: list[dict[str, float]] = []
    if isinstance(top3_raw, list):
        for item in top3_raw:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", "")).strip()
            try:
                p = float(item.get("p", 0.0))
            except Exception:  # noqa: BLE001 - user input parsing
                p = 0.0
            if label:
                top3.append({"label": label, "p": p})

    tier1: Tier1Result = {
        "category": category,
        "confidence": confidence,
        "top3": top3,
        "escalate": escalate,
    }
    return tier1, reasons


@lru_cache(maxsize=1)
def _get_server_tier1_runtime() -> dict[str, Any]:
    """Cache ONNX runtime objects for server-side Tier-1 fallback."""

    from CNN.shared import onnx_infer, preprocess  # local import (lazy)

    repo_root = Path(__file__).resolve().parents[2]
    onnx_path = repo_root / "CNN" / "models" / "tier1.onnx"
    label_map_path = repo_root / "CNN" / "models" / "label_map.json"
    infer_cfg_path = repo_root / "CNN" / "models" / "infer_config.json"

    label_map = json.loads(label_map_path.read_text(encoding="utf-8"))
    labels = preprocess.resolve_labels(label_map)
    preprocess.validate_labels(labels)

    infer_cfg = json.loads(infer_cfg_path.read_text(encoding="utf-8"))
    img_size = int(infer_cfg.get("img_size", 224))
    mean, std = preprocess.resolve_mean_std(infer_cfg)

    thresholds = {
        "conf": float(infer_cfg.get("conf_threshold", 0.75)),
        "margin": float(infer_cfg.get("margin_threshold", 0.15)),
        "topk": int(infer_cfg.get("topk", 3)),
        "reject_to_other": True,
    }

    return {
        "onnx": onnx_infer.OnnxInfer(onnx_path, prefer_cuda=True),
        "labels": labels,
        "img_size": img_size,
        "mean": mean,
        "std": std,
        "thresholds": thresholds,
    }


def _infer_tier1_on_server(img: Image.Image) -> tuple[Tier1Result, list[str]]:
    """Optional Tier-1 fallback inference on server (for QA / when tier1 not sent).

    We keep this lazy-imported to avoid importing onnxruntime unless needed.
    """

    from CNN.shared import decision as decision_utils  # local import (lazy)
    from CNN.shared import preprocess  # local import (lazy)

    rt = _get_server_tier1_runtime()
    tensor = preprocess.preprocess_image(
        img,
        img_size=int(rt["img_size"]),
        mean=rt["mean"],
        std=rt["std"],
    )
    logits = rt["onnx"].run(tensor)
    result = decision_utils.decide(logits, rt["labels"], rt["thresholds"])

    tier1: Tier1Result = {
        "category": str(result["top1_label"]),
        "confidence": float(result["top1_prob"]),
        "top3": result["top3"],
        "escalate": bool(result["escalate"]),
    }
    return tier1, list(result.get("reasons", []))


def _needs_followup(tier1: Tier1Result) -> bool:
    return bool(tier1.get("escalate")) or tier1.get("category") == "other_uncertain"


def _default_followup() -> dict[str, Any]:
    return {
        "needs_confirmation": True,
        "questions": [
            {
                "id": "q1",
                "type": "single_choice",
                "question": "Is there food residue or liquid inside?",
                "options": ["Yes", "No", "Not sure"],
            }
        ],
    }


app = FastAPI(title="BinItRight Scan Service", version=SCHEMA_VERSION)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/scan")
async def scan(
    image: UploadFile = File(...),
    tier1: Optional[str] = Form(default=None),
    timestamp: Optional[int] = Form(default=None),
) -> JSONResponse:
    t0 = time.perf_counter()
    request_id = str(uuid4())

    # 1) Read + validate the image (required by contract).
    try:
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes))
        # Avoid returning an Image bound to a closed fp.
        img.load()
    except Exception as exc:  # noqa: BLE001 - user input parsing
        return _error("INVALID_IMAGE", f"Invalid image: {exc}", status_code=400)

    # 2) Resolve Tier-1 result: prefer client-provided JSON, else server fallback.
    tier1_reasons: list[str] = []
    tier1_result: Optional[Tier1Result]
    if tier1:
        try:
            tier1_result, tier1_reasons = _parse_tier1_json(tier1)
        except Exception as exc:  # noqa: BLE001 - user input parsing
            return _error(
                "INVALID_TIER1",
                f"Invalid tier1 field: {exc}",
                status_code=400,
            )
    else:
        # v0.1 allows server-side Tier-1 fallback for parity/QA.
        try:
            tier1_result, tier1_reasons = _infer_tier1_on_server(img)
        except FileNotFoundError as exc:
            return _error("MODEL_NOT_FOUND", str(exc), status_code=500)
        except Exception as exc:  # noqa: BLE001 - inference failure
            return _error("TIER1_INFERENCE_FAILED", str(exc), status_code=500)

    assert tier1_result is not None

    # 3) Build decision + final mapping (Tier-2 is not used in v0.1).
    thresholds = _load_tier1_thresholds()
    reason_codes_set = set(map_reason_codes(tier1_reasons))

    conf_th = float(thresholds["conf_threshold"])
    margin_th = float(thresholds["margin_threshold"])
    if tier1_result["confidence"] < conf_th:
        reason_codes_set.add("LOW_CONFIDENCE")

    top3_sorted = sorted(tier1_result.get("top3", []), key=lambda x: x.get("p", 0.0), reverse=True)
    if len(top3_sorted) >= 2:
        margin = float(top3_sorted[0].get("p", 0.0)) - float(top3_sorted[1].get("p", 0.0))
        if margin < margin_th:
            reason_codes_set.add("LOW_MARGIN")

    if tier1_result["category"] == "other_uncertain":
        reason_codes_set.add("PRED_OTHER_UNCERTAIN")
    if tier1_result["escalate"] and not reason_codes_set:
        reason_codes_set.add("ESCALATE_FLAG")

    decision_obj: Dict[str, Any] = {
        "used_tier2": False,
        "reason_codes": sorted(reason_codes_set),
        "thresholds": thresholds,
    }

    if _needs_followup(tier1_result):
        # Safe default: do not over-claim a fine category when uncertain.
        final_input = dict(tier1_result)
        final_input["category"] = "other_uncertain"
        final_obj = final_from_tier1(final_input)  # type: ignore[arg-type]
        followup_obj = _default_followup()
    else:
        final_obj = final_from_tier1(tier1_result)
        followup_obj = None

    # 4) Assemble response.
    latency_ms = int(round((time.perf_counter() - t0) * 1000.0))
    data: Dict[str, Any] = {
        "tier1": tier1_result,
        "decision": decision_obj,
        "final": final_obj,
        "meta": {
            "schema_version": SCHEMA_VERSION,
            "latency_ms": {"total": latency_ms},
        },
    }
    if followup_obj is not None:
        data["followup"] = followup_obj
    if timestamp is not None:
        data["meta"]["timestamp"] = int(timestamp)

    payload: ScanResponse = {
        "status": "success",
        "request_id": request_id,
        "data": data,
    }
    return JSONResponse(status_code=200, content=payload)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
