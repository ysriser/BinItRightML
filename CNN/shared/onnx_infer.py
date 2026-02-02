"""ONNX Runtime helpers."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import onnxruntime as ort


def _pick_providers(prefer_cuda: bool = True) -> list[str]:
    available = ort.get_available_providers()
    if prefer_cuda and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def load_session(onnx_path: Path, prefer_cuda: bool = True) -> ort.InferenceSession:
    if not onnx_path.exists():
        raise FileNotFoundError(f"Missing ONNX model: {onnx_path}")
    providers = _pick_providers(prefer_cuda=prefer_cuda)
    return ort.InferenceSession(str(onnx_path), providers=providers)


class OnnxInfer:
    def __init__(self, onnx_path: Path, prefer_cuda: bool = True) -> None:
        self.session = load_session(onnx_path, prefer_cuda=prefer_cuda)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def run(self, batch: np.ndarray) -> np.ndarray:
        if batch.dtype != np.float32:
            batch = batch.astype(np.float32)
        outputs = self.session.run([self.output_name], {self.input_name: batch})
        return outputs[0]
