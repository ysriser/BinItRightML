import io
import json

from fastapi.testclient import TestClient
from PIL import Image

from CNN.services.scan_service_v0_1 import app


def _make_png_bytes() -> bytes:
    img = Image.new("RGB", (16, 16), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_scan_v0_1_envelope_with_tier1_payload():
    client = TestClient(app)
    image_bytes = _make_png_bytes()

    tier1 = {
        "category": "plastic",
        "confidence": 0.91,
        "top3": [
            {"label": "plastic", "p": 0.91},
            {"label": "glass", "p": 0.05},
            {"label": "other_uncertain", "p": 0.02},
        ],
        "escalate": False,
    }

    resp = client.post(
        "/api/v1/scan",
        files={"image": ("x.png", image_bytes, "image/png")},
        data={"tier1": json.dumps(tier1), "timestamp": "1730000000000"},
    )
    assert resp.status_code == 200
    body = resp.json()

    assert body["status"] == "success"
    assert isinstance(body["request_id"], str) and body["request_id"]

    data = body["data"]
    assert "decision" in data
    assert "final" in data

    final = data["final"]
    # Critical fields (v0.1)
    assert isinstance(final["category"], str) and final["category"]
    assert isinstance(final["recyclable"], bool)
    assert isinstance(final["confidence"], (int, float))
    assert isinstance(final["instruction"], str) and final["instruction"]


def test_scan_v0_1_uncertain_returns_followup():
    client = TestClient(app)
    image_bytes = _make_png_bytes()

    tier1 = {
        "category": "plastic",
        "confidence": 0.40,
        "top3": [
            {"label": "plastic", "p": 0.40},
            {"label": "paper", "p": 0.35},
            {"label": "other_uncertain", "p": 0.10},
        ],
        "escalate": True,
    }

    resp = client.post(
        "/api/v1/scan",
        files={"image": ("x.png", image_bytes, "image/png")},
        data={"tier1": json.dumps(tier1)},
    )
    assert resp.status_code == 200
    body = resp.json()

    assert body["status"] == "success"
    assert body["data"]["decision"]["used_tier2"] is False
    assert body["data"]["final"]["category"] == "Uncertain"
    assert body["data"]["followup"]["needs_confirmation"] is True
    assert isinstance(body["data"]["followup"]["questions"], list)

