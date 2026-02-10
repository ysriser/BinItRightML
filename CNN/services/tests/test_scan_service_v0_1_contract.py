import asyncio
import io
import json

import httpx
from PIL import Image

from CNN.services import scan_service_v0_1


def _make_image_bytes() -> bytes:
    image = Image.new("RGB", (64, 64), color=(100, 120, 140))
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return buf.getvalue()


def _post_scan(monkeypatch, *, tier1_payload: dict, force_cloud: bool = False):
    files = {
        "image": ("sample.jpg", _make_image_bytes(), "image/jpeg"),
        "tier1": (None, json.dumps(tier1_payload), "application/json"),
    }
    if force_cloud:
        files["force_cloud"] = (None, "true", "text/plain")

    async def _request():
        transport = httpx.ASGITransport(app=scan_service_v0_1.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post("/api/v1/scan", files=files)

    return asyncio.run(_request())


def test_scan_returns_final_five_fields_and_no_followup(monkeypatch):
    monkeypatch.setenv("TIER2_PROVIDER", "mock")

    response = _post_scan(
        monkeypatch,
        tier1_payload={
            "category": "other_uncertain",
            "confidence": 0.52,
            "top3": [
                {"label": "other_uncertain", "p": 0.52},
                {"label": "plastic", "p": 0.32},
                {"label": "paper", "p": 0.16},
            ],
            "escalate": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["status"] == "success"
    assert payload["data"]["decision"]["used_tier2"] is True

    final = payload["data"]["final"]
    assert set(final.keys()) == {
        "category",
        "recyclable",
        "confidence",
        "instruction",
        "instructions",
    }
    assert isinstance(final["instructions"], list)
    assert len(final["instructions"]) >= 2
    assert "followup" not in payload["data"]


async def _fake_openai_success(_img, _tier1):
    return {
        "category": "Ceramic mug",
        "recyclable": False,
        "confidence": 0.88,
        "instruction": "Dispose as general waste.",
        "instructions": [
            "Empty any liquid from the mug.",
            "If cracked, wrap before disposal.",
            "Dispose as general waste.",
        ],
    }


def test_force_cloud_openai_updates_all_final_fields(monkeypatch):
    monkeypatch.setenv("TIER2_PROVIDER", "openai")
    monkeypatch.setattr(scan_service_v0_1, "_openai_scan_final", _fake_openai_success)

    response = _post_scan(
        monkeypatch,
        tier1_payload={
            "category": "other_uncertain",
            "confidence": 0.91,
            "top3": [
                {"label": "other_uncertain", "p": 0.91},
                {"label": "glass", "p": 0.07},
                {"label": "paper", "p": 0.02},
            ],
            "escalate": True,
        },
        force_cloud=True,
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["data"]["meta"]["tier2_provider_used"] == "openai"
    final = payload["data"]["final"]
    assert final["category"] == "Ceramic mug"
    assert final["recyclable"] is False
    assert final["confidence"] == 0.88
    assert final["instruction"] == "Dispose as general waste."
    assert len(final["instructions"]) == 3


async def _fake_openai_fail(_img, _tier1):
    raise RuntimeError("simulated openai failure")


def test_openai_failure_falls_back_to_mock(monkeypatch):
    monkeypatch.setenv("TIER2_PROVIDER", "openai")
    monkeypatch.setattr(scan_service_v0_1, "_openai_scan_final", _fake_openai_fail)

    response = _post_scan(
        monkeypatch,
        tier1_payload={
            "category": "plastic",
            "confidence": 0.62,
            "top3": [
                {"label": "plastic", "p": 0.62},
                {"label": "glass", "p": 0.33},
                {"label": "paper", "p": 0.05},
            ],
            "escalate": False,
        },
        force_cloud=True,
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["data"]["meta"]["tier2_provider_attempted"] == "openai"
    assert payload["data"]["meta"]["tier2_provider_used"] == "mock"
    assert "TIER2_FALLBACK_MOCK" in payload["data"]["decision"]["reason_codes"]
    assert "tier2_error" in payload["data"]["meta"]