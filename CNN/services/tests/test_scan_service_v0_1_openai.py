from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
from PIL import Image

from CNN.services import scan_service_v0_1 as svc


def _make_image() -> Image.Image:
    return Image.new("RGB", (64, 64), color=(100, 120, 140))


def _tier1_payload() -> dict:
    return {
        "category": "other_uncertain",
        "confidence": 0.61,
        "top3": [
            {"label": "other_uncertain", "p": 0.61},
            {"label": "paper", "p": 0.29},
            {"label": "metal", "p": 0.10},
        ],
        "escalate": True,
    }


def test_error_helper_returns_expected_payload() -> None:
    response = svc._error("INVALID_IMAGE", "bad image", status_code=400)

    assert response.status_code == 400
    assert response.body


def test_load_thresholds_fallback_on_read_failure(monkeypatch) -> None:
    monkeypatch.setattr(Path, "read_text", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("no file")))

    thresholds = svc._load_tier1_thresholds()

    assert thresholds["conf_threshold"] == 0.70
    assert thresholds["margin_threshold"] == 0.05


def test_openai_scan_final_success_gpt5(monkeypatch) -> None:
    calls = []
    good_json = {
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": (
                            '{"category":"Ceramic mug","recyclable":false,'
                            '"confidence":0.77,"instruction":"Dispose as general waste.",' 
                            '"instructions":["Empty liquids.","Wrap broken edges if any."]}'
                        ),
                    }
                ],
            }
        ]
    }

    responses = [httpx.Response(200, request=httpx.Request("POST", "https://api.openai.com/v1/responses"), json=good_json)]

    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, headers=None, json=None):
            calls.append({"url": url, "headers": headers, "json": json})
            return responses.pop(0)

    monkeypatch.setattr(svc.httpx, "AsyncClient", lambda *args, **kwargs: FakeAsyncClient())
    monkeypatch.setattr(svc, "_get_openai_api_key", lambda: "fake-key")
    monkeypatch.setattr(svc, "_image_to_data_url", lambda img: "data:image/jpeg;base64,aaa")
    monkeypatch.setenv("OPENAI_TIER2_MODEL", "gpt-5-mini")
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "minimal")
    monkeypatch.setenv("OPENAI_VERBOSITY", "low")

    result = asyncio.run(
        svc._openai_scan_final(_make_image(), _tier1_payload(), timeout_s=2.0, max_retries=0)
    )

    assert result["category"] == "Ceramic mug"
    assert result["recyclable"] is False
    assert calls[0]["json"]["reasoning"]["effort"] == "minimal"
    assert calls[0]["json"]["text"]["verbosity"] == "low"


def test_openai_scan_final_success_non_gpt5_has_no_reasoning(monkeypatch) -> None:
    calls = []
    good_json = {
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": (
                            '{"category":"Paper cup","recyclable":false,'
                            '"confidence":0.66,"instruction":"Dispose as general waste.",' 
                            '"instructions":["Empty contents.","Dispose as general waste."]}'
                        ),
                    }
                ],
            }
        ]
    }

    responses = [httpx.Response(200, request=httpx.Request("POST", "https://api.openai.com/v1/responses"), json=good_json)]

    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, headers=None, json=None):
            calls.append(json)
            return responses.pop(0)

    monkeypatch.setattr(svc.httpx, "AsyncClient", lambda *args, **kwargs: FakeAsyncClient())
    monkeypatch.setattr(svc, "_get_openai_api_key", lambda: "fake-key")
    monkeypatch.setattr(svc, "_image_to_data_url", lambda img: "data:image/jpeg;base64,aaa")
    monkeypatch.setenv("OPENAI_TIER2_MODEL", "gpt-4o-mini")

    _ = asyncio.run(svc._openai_scan_final(_make_image(), _tier1_payload(), timeout_s=2.0, max_retries=0))

    assert "reasoning" not in calls[0]
    assert "verbosity" not in calls[0]["text"]


def test_openai_scan_final_retries_then_raises(monkeypatch) -> None:
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    responses = [
        httpx.Response(500, request=request, json={"error": {"message": "server error"}}),
        httpx.Response(400, request=request, json={"error": {"message": "bad request"}}),
    ]

    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, headers=None, json=None):
            return responses.pop(0)

    async def _fast_sleep(_seconds):
        return None

    monkeypatch.setattr(svc.httpx, "AsyncClient", lambda *args, **kwargs: FakeAsyncClient())
    monkeypatch.setattr(svc, "_get_openai_api_key", lambda: "fake-key")
    monkeypatch.setattr(svc, "_image_to_data_url", lambda img: "data:image/jpeg;base64,aaa")
    monkeypatch.setattr(svc.asyncio, "sleep", _fast_sleep)
    monkeypatch.setenv("OPENAI_TIER2_MODEL", "gpt-5-mini")

    try:
        asyncio.run(svc._openai_scan_final(_make_image(), _tier1_payload(), timeout_s=2.0, max_retries=1))
        assert False, "Expected RuntimeError"
    except RuntimeError as exc:
        assert "Tier-2 OpenAI call failed" in str(exc)
