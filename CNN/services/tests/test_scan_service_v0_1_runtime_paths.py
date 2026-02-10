from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from CNN.services import scan_service_v0_1 as svc


def test_load_strict_conf_thresholds_parses_valid_and_skips_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_read_text(self: Path, encoding: str = "utf-8") -> str:
        return json.dumps({"strict_conf_thresholds": {"plastic": "0.85", "broken": "x"}})

    monkeypatch.setattr(Path, "read_text", fake_read_text, raising=False)

    strict = svc._load_strict_conf_thresholds()

    assert strict["plastic"] == pytest.approx(0.85)
    assert "broken" not in strict


def test_load_strict_conf_thresholds_uses_defaults_when_file_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_missing(self: Path, encoding: str = "utf-8") -> str:
        raise FileNotFoundError("missing")

    monkeypatch.setattr(Path, "read_text", raise_missing, raising=False)

    strict = svc._load_strict_conf_thresholds()

    assert strict == {"plastic": 0.80, "glass": 0.80}


def test_get_server_tier1_runtime_builds_cached_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    from CNN.shared import onnx_infer, preprocess

    svc._get_server_tier1_runtime.cache_clear()

    class FakeOnnx:
        def __init__(self, path: Path, prefer_cuda: bool) -> None:
            self.path = str(path)
            self.prefer_cuda = prefer_cuda

    def fake_read_text(self: Path, encoding: str = "utf-8") -> str:
        path = str(self).replace("\\", "/")
        if path.endswith("label_map.json"):
            return json.dumps({"0": "paper", "1": "plastic"})
        if path.endswith("infer_config.json"):
            return json.dumps(
                {
                    "img_size": 256,
                    "mean": [0.1, 0.2, 0.3],
                    "std": [0.9, 0.8, 0.7],
                    "conf_threshold": 0.66,
                    "margin_threshold": 0.12,
                    "topk": 2,
                }
            )
        raise FileNotFoundError(path)

    monkeypatch.setattr(Path, "read_text", fake_read_text, raising=False)
    monkeypatch.setattr(onnx_infer, "OnnxInfer", FakeOnnx)
    monkeypatch.setattr(preprocess, "resolve_labels", lambda _label_map: ["paper", "plastic"])
    monkeypatch.setattr(preprocess, "validate_labels", lambda _labels: None)
    monkeypatch.setattr(preprocess, "resolve_mean_std", lambda _cfg: ([0.1, 0.2, 0.3], [0.9, 0.8, 0.7]))

    rt = svc._get_server_tier1_runtime()

    assert rt["img_size"] == 256
    assert rt["labels"] == ["paper", "plastic"]
    assert rt["thresholds"]["conf"] == pytest.approx(0.66)
    assert rt["thresholds"]["margin"] == pytest.approx(0.12)
    assert isinstance(rt["onnx"], FakeOnnx)

    svc._get_server_tier1_runtime.cache_clear()


def test_infer_tier1_on_server_maps_decision_result(monkeypatch: pytest.MonkeyPatch) -> None:
    from CNN.shared import decision as decision_utils
    from CNN.shared import preprocess

    class FakeOnnx:
        def run(self, _tensor: np.ndarray) -> np.ndarray:
            return np.array([[2.0, 1.0]], dtype=np.float32)

    runtime = {
        "onnx": FakeOnnx(),
        "labels": ["paper", "plastic"],
        "img_size": 224,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "thresholds": {"conf": 0.7, "margin": 0.1, "topk": 3, "reject_to_other": True},
    }

    monkeypatch.setattr(svc, "_get_server_tier1_runtime", lambda: runtime)
    monkeypatch.setattr(
        preprocess,
        "preprocess_image",
        lambda _img, img_size, mean, std: np.zeros((1, 3, img_size, img_size), dtype=np.float32),
    )
    monkeypatch.setattr(
        decision_utils,
        "decide",
        lambda _logits, _labels, _thresholds: {
            "top1_label": "paper",
            "top1_prob": 0.91,
            "top3": [{"label": "paper", "p": 0.91}],
            "escalate": False,
            "reasons": ["ok"],
        },
    )

    tier1, reasons = svc._infer_tier1_on_server(Image.new("RGB", (32, 32)))

    assert tier1["category"] == "paper"
    assert tier1["confidence"] == pytest.approx(0.91)
    assert tier1["escalate"] is False
    assert reasons == ["ok"]


def test_image_to_data_url_downscales_large_images() -> None:
    img = Image.new("RGB", (2000, 1000), color=(10, 20, 30))

    data_url = svc._image_to_data_url(img, max_side=500)

    assert data_url.startswith("data:image/jpeg;base64,")
    raw = base64.b64decode(data_url.split(",", 1)[1])
    with Image.open(io.BytesIO(raw)) as decoded:
        assert max(decoded.size) <= 500


def test_validate_llm_final_rejects_invalid_recyclable_and_instructions_type() -> None:
    with pytest.raises(ValueError, match="recyclable must be boolean"):
        svc._validate_llm_final(
            {
                "category": "Mug",
                "recyclable": "true",
                "confidence": 0.8,
                "instruction": "Dispose safely.",
                "instructions": ["Step 1", "Step 2"],
            }
        )

    with pytest.raises(ValueError, match="instructions must be an array"):
        svc._validate_llm_final(
            {
                "category": "Mug",
                "recyclable": False,
                "confidence": 0.8,
                "instruction": "Dispose safely.",
                "instructions": "bad",
            }
        )


def test_health_function_returns_ok() -> None:
    assert svc.health() == {"status": "ok"}
