import asyncio
import io
import json

import httpx
from PIL import Image

from CNN.services import scan_service_v0_1


def _make_image_bytes() -> bytes:
    image = Image.new("RGB", (64, 64), color=(90, 100, 110))
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return buf.getvalue()


def _post_scan(*, tier1_payload: dict):
    files = {
        "image": ("sample.jpg", _make_image_bytes(), "image/jpeg"),
        "tier1": (None, json.dumps(tier1_payload), "application/json"),
    }

    async def _request():
        transport = httpx.ASGITransport(app=scan_service_v0_1.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post("/api/v1/scan", files=files)

    return asyncio.run(_request())


def test_strict_class_threshold_triggers_tier2(monkeypatch):
    monkeypatch.setenv("TIER2_PROVIDER", "mock")

    response = _post_scan(
        tier1_payload={
            "category": "plastic",
            "confidence": 0.75,
            "top3": [
                {"label": "plastic", "p": 0.75},
                {"label": "paper", "p": 0.20},
                {"label": "glass", "p": 0.05},
            ],
            "escalate": False,
        }
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["data"]["decision"]["used_tier2"] is True
    assert "STRICT_CLASS_LOW_CONF" in payload["data"]["decision"]["reason_codes"]


def test_high_conf_non_strict_label_stays_tier1(monkeypatch):
    monkeypatch.setenv("TIER2_PROVIDER", "mock")

    response = _post_scan(
        tier1_payload={
            "category": "metal",
            "confidence": 0.76,
            "top3": [
                {"label": "metal", "p": 0.76},
                {"label": "plastic", "p": 0.18},
                {"label": "paper", "p": 0.06},
            ],
            "escalate": False,
        }
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["data"]["decision"]["used_tier2"] is False
    assert payload["data"]["final"]["category"] == "Metal"