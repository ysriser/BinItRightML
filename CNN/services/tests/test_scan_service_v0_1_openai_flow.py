from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
from PIL import Image

from CNN.services import scan_service_v0_1 as svc


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, Any] | None = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            req = httpx.Request("POST", "https://api.openai.com/v1/responses")
            resp = httpx.Response(self.status_code, request=req, text=self.text)
            raise httpx.HTTPStatusError("bad status", request=req, response=resp)

    def json(self) -> dict[str, Any]:
        return self._payload


def _success_payload(category: str = "Ceramic mug") -> dict[str, Any]:
    result = {
        "category": category,
        "recyclable": False,
        "confidence": 0.82,
        "instruction": "Dispose as general waste.",
        "instructions": [
            "Empty any remaining liquid.",
            "If broken, wrap before disposal.",
            "Dispose as general waste.",
        ],
    }
    return {
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": json.dumps(result)}],
            }
        ]
    }


def test_openai_scan_final_gpt5_includes_reasoning_and_verbosity(monkeypatch) -> None:
    posted: list[dict[str, Any]] = []

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, _url: str, *, headers: dict[str, str], json: dict[str, Any]):
            posted.append(json)
            assert headers["Authorization"].startswith("Bearer ")
            return _FakeResponse(200, _success_payload())

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_TIER2_MODEL", "gpt-5-mini")
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "minimal")
    monkeypatch.setenv("OPENAI_VERBOSITY", "low")
    monkeypatch.setattr(svc, "_image_to_data_url", lambda _img: "data:image/jpeg;base64,AAAA")
    monkeypatch.setattr(svc.httpx, "AsyncClient", FakeClient)

    tier1 = {"category": "other_uncertain", "confidence": 0.6, "top3": [], "escalate": True}
    result = asyncio.run(svc._openai_scan_final(Image.new("RGB", (32, 32)), tier1))

    assert result["category"] == "Ceramic mug"
    assert result["instruction"] == "Dispose as general waste."
    assert posted
    payload = posted[0]
    assert payload["reasoning"]["effort"] == "minimal"
    assert payload["text"]["verbosity"] == "low"


def test_openai_scan_final_non_gpt5_retries_on_server_error(monkeypatch) -> None:
    posted: list[dict[str, Any]] = []
    responses = [_FakeResponse(502), _FakeResponse(200, _success_payload("Glass cup"))]

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, _url: str, *, headers: dict[str, str], json: dict[str, Any]):
            posted.append(json)
            return responses.pop(0)

    async def _no_wait(_seconds: float) -> None:
        return None

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_TIER2_MODEL", "gpt-4o-mini")
    monkeypatch.setattr(svc, "_image_to_data_url", lambda _img: "data:image/jpeg;base64,BBBB")
    monkeypatch.setattr(svc.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(svc.asyncio, "sleep", _no_wait)

    tier1 = {"category": "glass", "confidence": 0.7, "top3": [], "escalate": False}
    result = asyncio.run(
        svc._openai_scan_final(
            Image.new("RGB", (40, 40)),
            tier1,
            max_retries=1,
        )
    )

    assert result["category"] == "Glass cup"
    assert len(posted) == 2
    assert "reasoning" not in posted[0]
    assert "verbosity" not in posted[0]["text"]


def test_openai_scan_final_raises_runtime_error_after_retries(monkeypatch) -> None:
    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, _url: str, *, headers: dict[str, str], json: dict[str, Any]):
            raise httpx.TimeoutException("timed out")

    async def _no_wait(_seconds: float) -> None:
        return None

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_TIER2_MODEL", "gpt-4o-mini")
    monkeypatch.setattr(svc, "_image_to_data_url", lambda _img: "data:image/jpeg;base64,CCCC")
    monkeypatch.setattr(svc.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(svc.asyncio, "sleep", _no_wait)

    tier1 = {"category": "paper", "confidence": 0.8, "top3": [], "escalate": False}
    try:
        asyncio.run(
            svc._openai_scan_final(
                Image.new("RGB", (48, 48)),
                tier1,
                max_retries=1,
            )
        )
        assert False, "Expected RuntimeError"
    except RuntimeError as exc:
        assert "Tier-2 OpenAI call failed" in str(exc)


def test_tier2_error_meta_handles_non_dict_error_payload() -> None:
    req = httpx.Request("POST", "https://api.openai.com/v1/responses")
    resp = httpx.Response(400, request=req, json={"error": "bad"})
    err = httpx.HTTPStatusError("bad", request=req, response=resp)

    meta = svc._tier2_error_meta(err)

    assert meta["http_status"] == "400"
    assert meta["code"] == "http_400"