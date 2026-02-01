import io
import os
import sys
import importlib

import torch
from fastapi.testclient import TestClient
from PIL import Image


class DummyModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        logits = torch.tensor([[0.1, 0.2, 2.0]], dtype=torch.float32)
        return logits.repeat(batch, 1)


def build_client() -> TestClient:
    # Force the service to skip real model loading during tests.
    os.environ["BINITRIGHT_SKIP_MODEL_LOAD"] = "1"
    module_name = "CNN.classification_6cats_new.serve"
    if module_name in sys.modules:
        del sys.modules[module_name]
    serve = importlib.import_module(module_name)

    def dummy_transform(_img: Image.Image) -> torch.Tensor:
        # Return a fixed-size tensor to mimic preprocessing.
        return torch.zeros(3, 224, 224)

    serve.STATE = {
        "model": DummyModel(),
        "labels": ["paper", "plastic", "glass"],
        "cfg": {"topk": 3, "conf_threshold": 0.75, "margin_threshold": 0.15, "backbone": "dummy"},
        "device": torch.device("cpu"),
        "transform": dummy_transform,
    }
    return TestClient(serve.app)


def test_health_endpoint():
    # Step 1: create a test client.
    client = build_client()
    # Step 2: call health endpoint.
    resp = client.get("/health")
    # Step 3: verify response.
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_scan_endpoint_returns_payload():
    # Step 1: create a test client.
    client = build_client()

    # Step 2: build a small in-memory JPEG to upload.
    img = Image.new("RGB", (64, 64), color=(120, 80, 40))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    # Step 3: send the image to the scan endpoint.
    resp = client.post("/api/v1/scan", files={"image": ("sample.jpg", buf.read(), "image/jpeg")})
    # Step 4: verify response payload shape.
    assert resp.status_code == 200
    payload = resp.json()

    assert payload["category"] == "glass"
    assert "confidence" in payload
    assert isinstance(payload["top3"], list)
    assert payload["instructions"]
