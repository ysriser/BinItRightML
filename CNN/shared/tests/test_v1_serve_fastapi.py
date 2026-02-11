from __future__ import annotations

import asyncio
import importlib
import io
import sys
from types import SimpleNamespace

import httpx
import numpy as np
from PIL import Image


class _DummyOnnxInfer:
    def __init__(self, _path):
        self.path = _path

    def run(self, batch):
        return np.array([[0.2, 0.8]], dtype=np.float32)


def _build_app(monkeypatch, *, escalate: bool = False):
    module_name = "CNN.experiments.v1_multicrop_reject.serve_fastapi_v1"
    if module_name in sys.modules:
        del sys.modules[module_name]

    fake_preprocess = SimpleNamespace(
        resolve_labels=lambda _map: ["paper", "other_uncertain"],
        validate_labels=lambda labels: None,
        resolve_mean_std=lambda _cfg: ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ensure_rgb=lambda img: img.convert("RGB"),
    )
    fake_multicrop = SimpleNamespace(
        run_multicrop=lambda **kwargs: (
            np.array([0.2, 0.8], dtype=np.float32),
            {"multicrop_used": True, "selected_crop": "center"},
        )
    )
    fake_decision = SimpleNamespace(
        decide_from_probs=lambda probs, labels, cfg: {
            "final_label": "other_uncertain" if escalate else "paper",
            "top1_prob": 0.8,
            "top3": [{"label": "paper", "p": 0.8}],
            "escalate": escalate,
            "reasons": ["LOW_CONFIDENCE"] if escalate else [],
            "metrics": {"max_prob": 0.8, "margin": 0.5, "entropy": 0.2},
        }
    )
    fake_onnx_infer = SimpleNamespace(OnnxInfer=_DummyOnnxInfer)

    monkeypatch.setitem(sys.modules, "CNN.shared.preprocess", fake_preprocess)
    monkeypatch.setitem(sys.modules, "CNN.shared.multicrop", fake_multicrop)
    monkeypatch.setitem(sys.modules, "CNN.shared.decision", fake_decision)
    monkeypatch.setitem(sys.modules, "CNN.shared.onnx_infer", fake_onnx_infer)

    import CNN.shared as shared_pkg

    monkeypatch.setattr(shared_pkg, "preprocess", fake_preprocess, raising=False)
    monkeypatch.setattr(shared_pkg, "multicrop", fake_multicrop, raising=False)
    monkeypatch.setattr(shared_pkg, "decision", fake_decision, raising=False)
    monkeypatch.setattr(shared_pkg, "onnx_infer", fake_onnx_infer, raising=False)

    module = importlib.import_module(module_name)
    return module


def _request(app, method: str, path: str, **kwargs):
    async def _run_request():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(_run_request())


def test_health_endpoint(monkeypatch) -> None:
    module = _build_app(monkeypatch, escalate=False)

    resp = _request(module.app, "GET", "/health")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_scan_endpoint_returns_payload(monkeypatch) -> None:
    module = _build_app(monkeypatch, escalate=False)

    img = Image.new("RGB", (64, 64), color=(120, 80, 40))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    resp = _request(
        module.app,
        "POST",
        "/api/v1/scan",
        files={"image": ("sample.jpg", buf.read(), "image/jpeg")},
    )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["category"] == "paper"
    assert payload["escalate"] is False
    assert isinstance(payload["instructions"], list)
    assert payload["debug"]["multicrop_used"] is True


def test_scan_endpoint_calls_expert_when_escalated(monkeypatch) -> None:
    module = _build_app(monkeypatch, escalate=True)

    called = {"count": 0}
    monkeypatch.setattr(
        module,
        "run_expert_vlm",
        lambda image_bytes, top3: called.__setitem__("count", called["count"] + 1),
    )

    img = Image.new("RGB", (64, 64), color=(120, 80, 40))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    resp = _request(
        module.app,
        "POST",
        "/api/v1/scan",
        files={"image": ("sample.jpg", buf.read(), "image/jpeg")},
    )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["category"] == "other_uncertain"
    assert payload["escalate"] is True
    assert called["count"] == 1


def test_scan_endpoint_rejects_invalid_image(monkeypatch) -> None:
    module = _build_app(monkeypatch, escalate=False)

    resp = _request(
        module.app,
        "POST",
        "/api/v1/scan",
        files={"image": ("bad.txt", b"not-an-image", "text/plain")},
    )

    assert resp.status_code == 400
    assert "Invalid image" in resp.text
