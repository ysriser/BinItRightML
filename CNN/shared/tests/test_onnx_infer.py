from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from CNN.shared import onnx_infer


class _NameObj:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeRuntimeSession:
    def __init__(self) -> None:
        self.last_batch = None

    def get_inputs(self):
        return [_NameObj("input_tensor")]

    def get_outputs(self):
        return [_NameObj("output_tensor")]

    def run(self, output_names, feeds):
        self.last_batch = feeds["input_tensor"]
        return [np.array([[1.0, 2.0]], dtype=np.float32)]


def test_pick_providers_prefers_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_ort = SimpleNamespace(
        get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    monkeypatch.setattr(onnx_infer, "ort", fake_ort, raising=False)

    providers = onnx_infer._pick_providers(prefer_cuda=True)

    assert providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_pick_providers_cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_ort = SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"])
    monkeypatch.setattr(onnx_infer, "ort", fake_ort, raising=False)

    providers = onnx_infer._pick_providers(prefer_cuda=True)

    assert providers == ["CPUExecutionProvider"]


def test_load_session_requires_onnxruntime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(onnx_infer, "ort", None, raising=False)

    with pytest.raises(RuntimeError, match="onnxruntime"):
        onnx_infer.load_session(Path("dummy.onnx"))


def test_load_session_missing_model_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_ort = SimpleNamespace(
        get_available_providers=lambda: ["CPUExecutionProvider"],
        InferenceSession=lambda path, providers: None,
    )
    monkeypatch.setattr(onnx_infer, "ort", fake_ort, raising=False)

    with pytest.raises(FileNotFoundError):
        onnx_infer.load_session(tmp_path / "missing.onnx")


def test_load_session_uses_selected_providers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"dummy")

    captured: dict[str, object] = {}

    class _DummySession:
        def __init__(self, path: str, providers: list[str]) -> None:
            captured["path"] = path
            captured["providers"] = providers

    fake_ort = SimpleNamespace(
        get_available_providers=lambda: ["CPUExecutionProvider"],
        InferenceSession=_DummySession,
    )
    monkeypatch.setattr(onnx_infer, "ort", fake_ort, raising=False)

    onnx_infer.load_session(onnx_path, prefer_cuda=False)

    assert captured["path"] == str(onnx_path)
    assert captured["providers"] == ["CPUExecutionProvider"]


def test_onnx_infer_run_casts_to_float32(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"dummy")
    fake = _FakeRuntimeSession()

    monkeypatch.setattr(onnx_infer, "load_session", lambda path, prefer_cuda=True: fake)

    runner = onnx_infer.OnnxInfer(onnx_path)
    batch = np.array([[1.0, 2.0]], dtype=np.float64)

    output = runner.run(batch)

    assert output.shape == (1, 2)
    assert fake.last_batch is not None
    assert fake.last_batch.dtype == np.float32
