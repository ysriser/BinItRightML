"""BinItRight Scan Service (v0.1 contract).

This FastAPI app implements the Android-facing endpoint defined in:
`CNN/docs/SCAN_SERVICE_SPEC_v0_1.md`

Goals for v0.1:
- Accept multipart upload: `image` (required), `tier1` (optional JSON string), `timestamp` (optional).
- Return a strict envelope: {status, request_id, data{decision, final, ...}}
- Tier-2 can be `mock` or `openai` (env: `TIER2_PROVIDER`).
  - provider=openai: LLM outputs `data.final` directly (strict JSON schema).
  - provider=mock OR OpenAI failure: fallback to deterministic templates.
- No quiz/followup questions are returned in v0.1 (Android should only depend on
  the 5 critical fields in `data.final`).

Run (from repo root):
  uvicorn CNN.services.scan_service_v0_1:app --host 0.0.0.0 --port 8000

Quick curl:
  curl -X POST http://127.0.0.1:8000/api/v1/scan -F "image=@test.jpg"
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

import httpx
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from CNN.shared.scan_contract_v0_1 import (
    ExpertDecision,
    ScanResponse,
    Tier1Result,
    ensure_special_prefix,
    final_from_expert_decision,
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
        "conf_threshold": float(infer_cfg.get("conf_threshold", 0.70)),
        "margin_threshold": float(infer_cfg.get("margin_threshold", 0.05)),
    }


def _load_strict_conf_thresholds() -> Dict[str, float]:
    """Optional per-class stricter confidence thresholds (server-side Tier-2 trigger).

    This is useful for classes that are costly to misclassify (e.g., plastic vs glass).

    Source of truth (if present): CNN/models/infer_config.json:
      strict_conf_thresholds: {"plastic": 0.8, "glass": 0.8}
    """

    infer_path = Path("CNN/models/infer_config.json")
    try:
        infer_cfg = json.loads(infer_path.read_text(encoding="utf-8"))
    except Exception:
        infer_cfg = {}

    strict_cfg = infer_cfg.get("strict_conf_thresholds", {})
    strict: Dict[str, float] = {}
    if isinstance(strict_cfg, dict):
        for k, v in strict_cfg.items():
            try:
                strict[str(k)] = float(v)
            except Exception:  # noqa: BLE001 - best effort only
                continue

    # Defaults if nothing is configured.
    if not strict:
        strict = {"plastic": 0.80, "glass": 0.80}
    return strict


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


def _parse_bool(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"true", "1", "yes", "y", "t"}


def _get_tier2_provider() -> str:
    provider = os.getenv("TIER2_PROVIDER", "mock").strip().lower()
    if provider not in {"mock", "openai"}:
        return "mock"
    return provider


def _get_openai_api_key() -> str:
    # Primary: OPENAI_API_KEY (recommended). Also accept LLM_API_KEY for local debugging.
    key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    if not key:
        raise RuntimeError("Missing OPENAI_API_KEY (or LLM_API_KEY) for Tier-2 provider=openai")
    return key


def _image_to_data_url(img: Image.Image, max_side: int = 768) -> str:
    """Downscale + encode as a JPEG data URL for OpenAI vision input."""

    img_rgb = img.convert("RGB")
    w, h = img_rgb.size
    scale = min(1.0, float(max_side) / float(max(w, h)))
    if scale < 1.0:
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        img_rgb = img_rgb.resize((new_w, new_h), resample=Image.BICUBIC)

    buf = io.BytesIO()
    img_rgb.save(buf, format="JPEG", quality=85, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _openai_final_schema() -> dict[str, Any]:
    # Keep schema simple: no advanced if/then to maximize compatibility.
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["category", "recyclable", "confidence", "instruction", "instructions"],
        "properties": {
            "category": {
                "type": "string",
                "minLength": 1,
                "maxLength": 60,
                "description": (
                    "User-facing refined item name. MUST start with 'E-waste - ' if e-waste, "
                    "or 'Textile - ' if textile."
                ),
            },
            "recyclable": {
                "type": "boolean",
                "description": (
                    "True only if it should go to the normal recycling bin (blue bin flow). "
                    "E-waste/Textile should be false."
                ),
            },
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "instruction": {"type": "string", "minLength": 1, "maxLength": 120},
            "instructions": {
                "type": "array",
                "minItems": 2,
                "maxItems": 8,
                "items": {"type": "string", "minLength": 1, "maxLength": 140},
            },
        },
    }


def _extract_responses_output_text(resp_json: dict[str, Any]) -> str:
    output = resp_json.get("output", [])
    texts: list[str] = []
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "message":
                continue
            for content in item.get("content", []) or []:
                if isinstance(content, dict) and content.get("type") == "output_text":
                    texts.append(str(content.get("text", "")))
    text = "".join(texts).strip()
    if not text:
        raise ValueError("OpenAI response did not contain output_text")
    return text


def _validate_llm_final(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("ScanFinal must be a JSON object")

    allowed_keys = {"category", "recyclable", "confidence", "instruction", "instructions"}
    if set(raw.keys()) != allowed_keys:
        raise ValueError("ScanFinal must contain exactly the 5 required fields")

    category = str(raw.get("category", "")).strip()
    if not category:
        raise ValueError("category is required")

    recyclable = raw.get("recyclable")
    if not isinstance(recyclable, bool):
        raise ValueError("recyclable must be boolean")

    try:
        confidence = float(raw.get("confidence", 0.0))
    except Exception as exc:  # noqa: BLE001 - LLM output parsing
        raise ValueError(f"confidence must be number: {exc}") from exc
    confidence = max(0.0, min(1.0, confidence))

    instruction = str(raw.get("instruction", "")).strip()
    if not instruction:
        raise ValueError("instruction is required")

    instructions_raw = raw.get("instructions", [])
    if not isinstance(instructions_raw, list):
        raise ValueError("instructions must be an array")
    instructions: list[str] = [str(x).strip() for x in instructions_raw if str(x).strip()]
    if len(instructions) < 2:
        raise ValueError("instructions must contain at least 2 non-empty steps")
    if len(instructions) > 8:
        instructions = instructions[:8]

    # Policy: never provide scanning/photo/camera tips; only disposal instructions.
    combined = (" ".join([instruction] + instructions)).lower()
    banned = [

    ]
    if any(b in combined for b in banned):
        raise ValueError("ScanFinal must not include scanning/camera tips")

    return {
        "category": category,
        "recyclable": recyclable,
        "confidence": confidence,
        "instruction": instruction,
        "instructions": instructions,
    }


async def _openai_scan_final(
    img: Image.Image,
    tier1: Tier1Result,
    *,
    timeout_s: float = 5.0,
    max_retries: int = 0,
) -> dict[str, Any]:
    api_key = _get_openai_api_key()
    model = os.getenv("OPENAI_TIER2_MODEL", "gpt-5-mini")
    is_gpt5 = "gpt-5" in model.lower()
    # GPT-5 knobs (for lower latency). Other models may reject unknown fields.
    reasoning_effort = os.getenv("OPENAI_REASONING_EFFORT", "minimal")
    verbosity = os.getenv("OPENAI_VERBOSITY", "low")

    image_url = _image_to_data_url(img)
    tier1_json = json.dumps(tier1, ensure_ascii=False)

    developer_msg = (
    "You are a recycling disposal expert. Given ONE photo of a waste item (Singapore context, "
    "do not cite specific shop names or addresses), output ONLY a JSON object that matches the "
    "provided schema. No extra text. Keep it short and precise.\n\n"
    "Rules:\n"
    "1) Do NOT ask the user questions. Do NOT output quiz or follow-up questions.\n"
    "2) category:\n"
    "   - If it is e-waste (electronics, battery, cables, chargers, small devices), category MUST "
    "start with 'E-waste - '.\n"
    "   - If it is textile/fabric/clothing, category MUST start with 'Textile - '.\n"
    "   - Otherwise use a short refined name (for example: 'PET Plastic Bottle', 'Glass Container', "
    "'Oily Pizza Box', 'Takeaway Drink Cup with Straw').\n"
    "3) recyclable:\n"
    "   - true ONLY for normal recycling bin flow (clean, dry, single-material recyclable).\n"
    "   - false for e-waste, textile, contaminated paper, heavily food-stained items, or unknown items.\n"
    "4) instructions:\n"
    "   - Provide actionable disposal steps (2-5), imperative style.\n"
    "   - If multiple parts exist, include disposal guidance for each part.\n"
    "   - For mixed items, clearly explain cleaning/dismantling and where each part goes.\n"
    "   - If special drop-off is needed, say generic 'bring to an e-waste recycling point'.\n"
    "5) confidence:\n"
    "   - 0.85-0.99 when very clear; 0.55-0.80 when somewhat clear.\n"
    "   - <=0.54 when uncertain: set category to 'Uncertain', recyclable=false, and provide "
    "conservative disposal guidance (general waste) with safety notes for batteries/e-waste.\n"
    "6) Never invent a confident wrong class. If not sure, return 'Uncertain'.\n"
    "7) Off-topic/fake photos are possible. If image is not a real waste item, return 'Uncertain' "
    "with safe generic disposal guidance."
)
user_msg = (
        "Return the final scan result JSON.\n"
        f"Tier-1 context (may be wrong): {tier1_json}"
    )

    payload: dict[str, Any] = {
        "model": model,
        # Keep Tier-2 fast and bounded; the schema already limits lengths.
        "max_output_tokens": 256,
        "input": [
            {
                "role": "developer",
                "content": [{"type": "input_text", "text": developer_msg}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_msg},
                    {"type": "input_image", "image_url": image_url, "detail": "low"},
                ],
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "scan_final",
                "strict": True,
                "schema": _openai_final_schema(),
            }
        },
    }
    if is_gpt5:
        payload["reasoning"] = {"effort": reasoning_effort}
        payload["text"]["verbosity"] = verbosity

    url = "https://api.openai.com/v1/responses"
    # Keep headers explicit; OpenAI may ignore unknown headers, but this helps
    # with older/staged deployments of the Responses endpoint.
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "OpenAI-Beta": "responses=v1",
    }

    last_exc: Optional[Exception] = None
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_s)) as client:
        for attempt in range(max_retries + 1):
            try:
                resp = await client.post(url, headers=headers, json=payload)
                if resp.status_code >= 500 and attempt < max_retries:
                    await asyncio.sleep(0.25)
                    continue
                resp.raise_for_status()
                resp_json = resp.json()
                text = _extract_responses_output_text(resp_json)
                raw = json.loads(text)
                return _validate_llm_final(raw)
            except Exception as exc:  # noqa: BLE001 - network/parse failures
                last_exc = exc
                if attempt < max_retries:
                    await asyncio.sleep(0.25)
                    continue
                break

    raise RuntimeError("Tier-2 OpenAI call failed") from last_exc


def _mock_tier2_decision(
    tier1: Tier1Result,
    *,
    force_cloud: bool,
) -> ExpertDecision:
    """Return a deterministic, template-friendly "expert decision" (mock Tier-2).

    This intentionally avoids long free-text. It outputs a small structured decision
    and leaves instruction composition to `final_from_expert_decision()`.
    """

    label = str(tier1.get("category", "other_uncertain"))

    # (a) disposal stream / bin type (ExpertDecision enum)
    if label in {"paper", "plastic", "metal", "glass"}:
        bin_type: str = "recyclable"
    elif label == "textile":
        bin_type = "textile"
    elif label == "e-waste":
        bin_type = "e_waste"
    else:
        # Safe fallback for unknown/composite items.
        bin_type = "non_recyclable"

    # (b) contamination flag heuristic
    contamination: str = "unknown"
    if force_cloud and not bool(tier1.get("escalate")) and label != "other_uncertain":
        contamination = "clean"

    # (c) optional refined display name
    refined_name = ""
    if contamination == "clean":
        if label == "plastic":
            refined_name = "Plastic Container"
        elif label == "glass":
            refined_name = "Glass Container"

    hazardous = bin_type == "e_waste"
    needs_confirmation = contamination == "unknown" or hazardous or label == "other_uncertain"

    expert: ExpertDecision = {
        "bin_type": bin_type,  # type: ignore[typeddict-item]
        "contamination_flag": contamination,  # type: ignore[typeddict-item]
        "needs_confirmation": needs_confirmation,
    }
    if refined_name:
        expert["refined_name"] = refined_name

    return expert


def _final_from_llm_min(
    llm_final: dict[str, Any],
) -> Dict[str, Any]:
    """Compose server `final` object from LLM's 5-field output.

    v3.3: return ONLY the 5 UI-required fields. Legacy fields are omitted; Android
    routes special streams using (category prefix + recyclable).
    """

    category = str(llm_final["category"]).strip()
    # Trust Tier-2 (LLM) for special-stream routing. Only normalize prefixes if present.
    if category.lower().startswith("e-waste"):
        category = ensure_special_prefix(category, "e-waste")
    if category.lower().startswith("textile"):
        category = ensure_special_prefix(category, "textile")

    recyclable = bool(llm_final["recyclable"])
    if category.lower().startswith(("e-waste - ", "textile - ")):
        # v0.1 meaning: these are not blue-bin flow.
        recyclable = False

    confidence = float(llm_final["confidence"])
    instruction = str(llm_final["instruction"]).strip()
    instructions = list(llm_final["instructions"])

    return {
        "category": category,
        "recyclable": recyclable,
        "confidence": confidence,
        "instruction": instruction,
        "instructions": instructions,
    }


def _extract_openai_error_json(resp: httpx.Response) -> dict[str, Any]:
    try:
        payload = resp.json()
    except Exception:  # noqa: BLE001 - best effort only
        return {}
    return payload if isinstance(payload, dict) else {}


def _tier2_error_meta(exc: Exception) -> Dict[str, str]:
    """Return compact, non-secret error info for meta.tier2_error.

    Contract:
      {"http_status": "...", "code": "...", "message": "..."}
    """

    root = exc
    if isinstance(exc, RuntimeError) and exc.__cause__ is not None:
        root = exc.__cause__

    msg_l = str(root).lower()
    if isinstance(root, RuntimeError) and "missing openai_api_key" in msg_l:
        return {
            "http_status": "",
            "code": "missing_api_key",
            "message": "Missing OPENAI_API_KEY/LLM_API_KEY",
        }
    if isinstance(root, (TimeoutError, httpx.TimeoutException)):
        return {"http_status": "", "code": "timeout", "message": "OpenAI request timed out"}

    if isinstance(root, httpx.HTTPStatusError):
        status = int(root.response.status_code)
        payload = _extract_openai_error_json(root.response)
        err = payload.get("error", {})
        if not isinstance(err, dict):
            err = {}
        err_type = str(err.get("type", "")).strip()
        err_code = str(err.get("code", "")).strip()
        err_msg = str(err.get("message", "")).strip()

        code = err_code or err_type or f"http_{status}"
        if err_msg:
            message = err_msg
        else:
            # Best-effort: include a short snippet to help diagnose non-JSON errors
            # (e.g., proxies returning HTML). Never include secrets.
            snippet = (root.response.text or "").strip().replace("\n", " ")
            message = snippet[:200] if snippet else "OpenAI request failed"
        return {"http_status": str(status), "code": code, "message": message}

    if isinstance(root, httpx.RequestError):
        return {"http_status": "", "code": "network", "message": root.__class__.__name__}
    if isinstance(root, ValueError):
        msg = str(root).strip() or root.__class__.__name__
        return {"http_status": "", "code": "schema", "message": msg[:200]}

    return {"http_status": "", "code": "unknown", "message": root.__class__.__name__}


app = FastAPI(title="BinItRight Scan Service", version=SCHEMA_VERSION)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/scan")
async def scan(
    image: UploadFile = File(...),
    tier1: Optional[str] = Form(default=None),
    timestamp: Optional[int] = Form(default=None),
    force_cloud: Optional[str] = Form(default=None),
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

    # 3) Build decision + final mapping (mock Tier-2 in v0.1).
    thresholds = _load_tier1_thresholds()
    strict_conf = _load_strict_conf_thresholds()
    # Expose per-class thresholds in decision.thresholds for observability (Android ignores).
    if "plastic" in strict_conf:
        thresholds["conf_threshold_plastic"] = float(strict_conf["plastic"])
    if "glass" in strict_conf:
        thresholds["conf_threshold_glass"] = float(strict_conf["glass"])

    reason_codes_set = set(map_reason_codes(tier1_reasons))
    force_cloud_bool = _parse_bool(force_cloud)

    conf_th = float(thresholds["conf_threshold"])
    label = str(tier1_result.get("category", "other_uncertain"))
    effective_conf_th = float(max(conf_th, strict_conf.get(label, conf_th)))
    margin_th = float(thresholds["margin_threshold"])
    if tier1_result["confidence"] < conf_th:
        reason_codes_set.add("LOW_CONFIDENCE")
    if effective_conf_th > conf_th and tier1_result["confidence"] < effective_conf_th:
        reason_codes_set.add("STRICT_CLASS_LOW_CONF")

    top3_sorted = sorted(
        tier1_result.get("top3", []),
        key=lambda x: x.get("p", 0.0),
        reverse=True,
    )
    low_margin_trigger = False
    if len(top3_sorted) >= 2:
        margin = float(top3_sorted[0].get("p", 0.0)) - float(
            top3_sorted[1].get("p", 0.0)
        )
        if margin < margin_th:
            reason_codes_set.add("LOW_MARGIN")
            low_margin_trigger = True

    if tier1_result["category"] == "other_uncertain":
        reason_codes_set.add("PRED_OTHER_UNCERTAIN")
    if bool(tier1_result.get("escalate")):
        reason_codes_set.add("TIER1_ESCALATE")
    if force_cloud_bool:
        reason_codes_set.add("FORCE_CLOUD")

    low_conf_trigger = tier1_result["confidence"] < effective_conf_th
    trigger_tier2 = (
        force_cloud_bool
        or bool(tier1_result.get("escalate"))
        or tier1_result.get("category") == "other_uncertain"
        or low_conf_trigger
        or low_margin_trigger
    )

    decision_obj: Dict[str, Any] = {
        "used_tier2": bool(trigger_tier2),
        "reason_codes": sorted(reason_codes_set),
        "thresholds": thresholds,
    }

    expert_obj: Optional[ExpertDecision] = None
    tier2_provider_attempted: Optional[str] = None
    tier2_provider_used: Optional[str] = None
    tier2_error: Optional[Dict[str, str]] = None
    if trigger_tier2:
        provider_requested = _get_tier2_provider()
        tier2_provider_attempted = provider_requested
        provider_used = provider_requested
        tier2_provider_used = provider_used

        llm_final: Optional[dict[str, Any]] = None
        if provider_requested == "openai":
            try:
                llm_final = await _openai_scan_final(img, tier1_result)
            except Exception as exc:  # noqa: BLE001 - fallback required
                reason_codes_set.add("TIER2_FALLBACK_MOCK")
                tier2_error = _tier2_error_meta(exc)
                provider_used = "mock"
                tier2_provider_used = provider_used

        if provider_used == "openai" and llm_final is not None:
            final_obj = _final_from_llm_min(llm_final)
        else:
            expert_obj = _mock_tier2_decision(tier1_result, force_cloud=force_cloud_bool)
            # Safety: never expose any quiz/questions fields.
            expert_obj.pop("questions", None)
            final_obj = final_from_expert_decision(tier1_result, expert_obj)
    else:
        final_obj = final_from_tier1(tier1_result)

    # reason_codes_set can be updated during Tier-2 (fallback). Keep decision in sync.
    decision_obj["reason_codes"] = sorted(reason_codes_set)

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
    if trigger_tier2:
        data["meta"]["force_cloud"] = bool(force_cloud_bool)
        if tier2_provider_attempted is not None:
            data["meta"]["tier2_provider_attempted"] = tier2_provider_attempted
        if tier2_provider_used is not None:
            data["meta"]["tier2_provider_used"] = tier2_provider_used
            data["meta"]["tier2_provider"] = tier2_provider_used  # short alias for logs
        if tier2_error is not None:
            data["meta"]["tier2_error"] = tier2_error
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

