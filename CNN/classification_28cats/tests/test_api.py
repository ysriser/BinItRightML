import io
from pathlib import Path

import torch
from fastapi.testclient import TestClient
from PIL import Image

import serve


class DummyModel(torch.nn.Module):
    """Simple model stub that returns fixed logits for deterministic tests."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        logits = torch.tensor([[2.0, 1.0, 0.5]], dtype=torch.float32)
        return logits.repeat(batch, 1)


def _build_app(tmp_path: Path, monkeypatch) -> TestClient:
    # Minimal config to drive the API behavior in tests.
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "confidence_threshold: 0.6\nhigh_risk_categories: [glass]\n",
        encoding="utf-8",
    )

    meta = {
        "class_names": ["plastic", "glass", "paper"],
        "model_version": "test",
        "backbone": "efficientnet_b0",
        "image_size": 224,
    }

    def fake_load_artifacts(artifact: Path, labels: Path, device: torch.device):
        def eval_tfm(pil_image: Image.Image) -> torch.Tensor:
            # Return a dummy tensor with a stable shape.
            return torch.zeros(3, 224, 224)

        return DummyModel(), meta, eval_tfm

    monkeypatch.setattr(serve, "load_artifacts", fake_load_artifacts)
    app = serve.create_app(Path("artifact"), Path("labels"), cfg_path, torch.device("cpu"))
    return TestClient(app)


def test_health_endpoint(tmp_path: Path, monkeypatch):
    client = _build_app(tmp_path, monkeypatch)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_scan_endpoint_success(tmp_path: Path, monkeypatch):
    client = _build_app(tmp_path, monkeypatch)

    # Build a small in-memory JPEG for the upload.
    img = Image.new("RGB", (64, 64), color=(120, 80, 40))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/api/v1/scan",
        files={"image": ("sample.jpg", buf.read(), "image/jpeg")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["category"] in {"plastic", "glass", "paper"}
    assert "confidence" in payload
    assert "instructions" in payload
    assert "top3" in payload


def test_scan_endpoint_invalid_image(tmp_path: Path, monkeypatch):
    client = _build_app(tmp_path, monkeypatch)
    response = client.post(
        "/api/v1/scan",
        files={"image": ("bad.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400
