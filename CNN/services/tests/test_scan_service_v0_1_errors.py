from __future__ import annotations

import asyncio
import io
import json

import httpx
from PIL import Image

from CNN.services import scan_service_v0_1 as svc


def _image_bytes() -> bytes:
    img = Image.new("RGB", (48, 48), color=(120, 110, 100))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _post_scan(*, tier1_payload: dict | None = None, force_cloud: bool = False, bad_image: bool = False, timestamp: int | None = None):
    files: dict[str, tuple[str | None, object, str]] = {
        "image": ("x.jpg", b"not-an-image" if bad_image else _image_bytes(), "image/jpeg"),
    }
    if tier1_payload is not None:
        files["tier1"] = (None, json.dumps(tier1_payload), "application/json")
    if force_cloud:
        files["force_cloud"] = (None, "true", "text/plain")
    if timestamp is not None:
        files["timestamp"] = (None, str(timestamp), "text/plain")

    async def _request():
        transport = httpx.ASGITransport(app=svc.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post("/api/v1/scan", files=files)

    return asyncio.run(_request())


def test_parse_tier1_json_invalid_cases() -> None:
    try:
        svc._parse_tier1_json("{bad-json")
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "Invalid tier1 JSON" in str(exc)

    try:
        svc._parse_tier1_json("[]")
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "must be a JSON object" in str(exc)


def test_parse_tier1_json_unknown_label_and_invalid_confidence() -> None:
    tier1, reasons = svc._parse_tier1_json(
        json.dumps(
            {
                "category": "mug",
                "confidence": "bad",
                "top3": [{"label": "mug", "p": "oops"}],
                "escalate": 1,
            }
        )
    )

    assert tier1 is not None
    assert tier1["category"] == "other_uncertain"
    assert tier1["confidence"] == 0.0
    assert tier1["escalate"] is True
    assert "unknown_tier1_label" in reasons
    assert "invalid_tier1_confidence" in reasons


def test_scan_invalid_image_returns_400() -> None:
    response = _post_scan(bad_image=True)

    assert response.status_code == 400
    body = response.json()
    assert body["code"] == "INVALID_IMAGE"


def test_scan_invalid_tier1_json_returns_400() -> None:
    files = {
        "image": ("x.jpg", _image_bytes(), "image/jpeg"),
        "tier1": (None, "{bad-json", "application/json"),
    }

    async def _request():
        transport = httpx.ASGITransport(app=svc.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post("/api/v1/scan", files=files)

    response = asyncio.run(_request())

    assert response.status_code == 400
    body = response.json()
    assert body["code"] == "INVALID_TIER1"


def test_scan_without_tier1_model_not_found(monkeypatch) -> None:
    monkeypatch.setattr(svc, "_infer_tier1_on_server", lambda _img: (_ for _ in ()).throw(FileNotFoundError("no model")))

    response = _post_scan(tier1_payload=None)

    assert response.status_code == 500
    assert response.json()["code"] == "MODEL_NOT_FOUND"


def test_scan_without_tier1_inference_failed(monkeypatch) -> None:
    monkeypatch.setattr(svc, "_infer_tier1_on_server", lambda _img: (_ for _ in ()).throw(RuntimeError("boom")))

    response = _post_scan(tier1_payload=None)

    assert response.status_code == 500
    assert response.json()["code"] == "TIER1_INFERENCE_FAILED"


def test_scan_non_tier2_path_keeps_tier1_and_timestamp() -> None:
    response = _post_scan(
        tier1_payload={
            "category": "metal",
            "confidence": 0.95,
            "top3": [
                {"label": "metal", "p": 0.95},
                {"label": "paper", "p": 0.03},
                {"label": "glass", "p": 0.02},
            ],
            "escalate": False,
        },
        timestamp=1730000000000,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["data"]["decision"]["used_tier2"] is False
    assert payload["data"]["meta"]["timestamp"] == 1730000000000
    assert "tier2_provider_used" not in payload["data"]["meta"]


def test_scan_low_margin_triggers_tier2() -> None:
    response = _post_scan(
        tier1_payload={
            "category": "metal",
            "confidence": 0.90,
            "top3": [
                {"label": "metal", "p": 0.90},
                {"label": "paper", "p": 0.89},
                {"label": "glass", "p": 0.01},
            ],
            "escalate": False,
        }
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["data"]["decision"]["used_tier2"] is True
    assert "LOW_MARGIN" in payload["data"]["decision"]["reason_codes"]


def test_openai_final_schema_has_required_keys() -> None:
    schema = svc._openai_final_schema()

    assert schema["type"] == "object"
    assert "category" in schema["properties"]
    assert "instructions" in schema["properties"]


def test_tier2_error_meta_additional_branches() -> None:
    req_err = svc._tier2_error_meta(httpx.RequestError("offline"))
    assert req_err["code"] == "network"

    value_err = svc._tier2_error_meta(ValueError("bad schema"))
    assert value_err["code"] == "schema"

    unknown_err = svc._tier2_error_meta(Exception("x"))
    assert unknown_err["code"] == "unknown"

    req = httpx.Request("POST", "https://api.openai.com/v1/responses")
    resp = httpx.Response(502, request=req, text="<html>gateway</html>")
    http_err = httpx.HTTPStatusError("bad", request=req, response=resp)
    meta = svc._tier2_error_meta(http_err)
    assert meta["http_status"] == "502"
    assert meta["code"] == "http_502"
