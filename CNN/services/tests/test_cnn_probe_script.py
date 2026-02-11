from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROBE_SCRIPT = Path(__file__).resolve().parents[2] / "test.py"
pytestmark = pytest.mark.skipif(
    not PROBE_SCRIPT.exists(),
    reason="CNN/test.py removed; probe script tests skipped",
)


class FakeResponse:
    def __init__(self, status_code: int, payload=None, text: str = "", raise_json: bool = False):
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self._raise_json = raise_json

    def json(self):
        if self._raise_json:
            raise ValueError("bad json")
        return self._payload


def load_probe_module(monkeypatch: pytest.MonkeyPatch):
    script = PROBE_SCRIPT
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key")

    # Ensure import does not require real third-party requests package.
    fake_requests = SimpleNamespace(
        get=lambda *args, **kwargs: FakeResponse(200, payload={"data": []}),
        post=lambda *args, **kwargs: FakeResponse(200, payload={"id": "x", "output": []}),
    )
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    name = "cnn_test_probe_module"
    if name in sys.modules:
        del sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_call_models_list_success(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_probe_module(monkeypatch)

    monkeypatch.setattr(
        module.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(
            200,
            payload={"data": [{"id": "gpt-a"}, {"id": "gpt-b"}, {"x": "ignore"}]},
        ),
    )

    visible = module.call_models_list()

    assert visible == {"gpt-a", "gpt-b"}


def test_call_models_list_failure_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_probe_module(monkeypatch)

    monkeypatch.setattr(
        module.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(401, text="unauthorized", raise_json=True),
    )

    assert module.call_models_list() is None


def test_call_responses_text_handles_success(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    module = load_probe_module(monkeypatch)

    monkeypatch.setattr(
        module.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(
            200,
            payload={
                "id": "resp_1",
                "output": [{"content": [{"text": "OK"}]}],
            },
        ),
    )

    module.call_responses_text("gpt-5-mini")
    out = capsys.readouterr().out

    assert "HTTP 200" in out
    assert "response_id: resp_1" in out


def test_call_responses_text_handles_error(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    module = load_probe_module(monkeypatch)

    monkeypatch.setattr(
        module.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(404, payload={"error": {"code": "model_not_found"}}),
    )

    module.call_responses_text("bad-model")
    out = capsys.readouterr().out

    assert "HTTP 404" in out


def test_call_responses_with_image_missing_file_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_probe_module(monkeypatch)

    with pytest.raises(SystemExit, match="Image not found"):
        module.call_responses_with_image("gpt-5-mini", "missing.png")


def test_call_responses_with_image_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    module = load_probe_module(monkeypatch)

    img = tmp_path / "x.png"
    img.write_bytes(b"\x89PNG\r\n")

    monkeypatch.setattr(
        module.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(
            200,
            payload={
                "id": "resp_img",
                "output": [{"content": [{"text": "A bottle"}]}],
            },
        ),
    )

    module.call_responses_with_image("gpt-5-mini", str(img))
    out = capsys.readouterr().out

    assert "HTTP 200" in out
    assert "response_id: resp_img" in out


def test_module_without_key_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    script = PROBE_SCRIPT
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(get=lambda *a, **k: None, post=lambda *a, **k: None))

    name = "cnn_test_probe_no_key"
    if name in sys.modules:
        del sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)

    with pytest.raises(SystemExit, match="Missing OPENAI_API_KEY"):
        spec.loader.exec_module(module)

