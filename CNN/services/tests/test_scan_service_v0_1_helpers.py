from __future__ import annotations


import httpx
import pytest

from CNN.services import scan_service_v0_1 as svc


def test_parse_bool_accepts_common_true_values() -> None:
    assert svc._parse_bool("true") is True
    assert svc._parse_bool(" 1 ") is True
    assert svc._parse_bool("YES") is True
    assert svc._parse_bool(None) is False
    assert svc._parse_bool("no") is False


def test_get_tier2_provider_fallbacks_to_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TIER2_PROVIDER", "invalid")
    assert svc._get_tier2_provider() == "mock"

    monkeypatch.setenv("TIER2_PROVIDER", "openai")
    assert svc._get_tier2_provider() == "openai"


def test_get_openai_api_key_supports_fallback_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "abc")

    assert svc._get_openai_api_key() == "abc"


def test_get_openai_api_key_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        svc._get_openai_api_key()


def test_extract_responses_output_text_parses_message() -> None:
    payload = {
        "output": [
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "{"},
                    {"type": "output_text", "text": '"a":1}'},
                ],
            }
        ]
    }

    text = svc._extract_responses_output_text(payload)

    assert text == '{"a":1}'


def test_extract_responses_output_text_raises_when_empty() -> None:
    with pytest.raises(ValueError, match="output_text"):
        svc._extract_responses_output_text({"output": []})


def test_validate_llm_final_accepts_valid_payload() -> None:
    raw = {
        "category": "Metal Can",
        "recyclable": True,
        "confidence": 0.88,
        "instruction": "Empty the can.",
        "instructions": ["Empty the can.", "Rinse quickly.", "Put it in blue bin."],
    }

    validated = svc._validate_llm_final(raw)

    assert validated["category"] == "Metal Can"
    assert validated["recyclable"] is True
    assert validated["confidence"] == 0.88


def test_validate_llm_final_rejects_wrong_keys() -> None:
    raw = {
        "category": "A",
        "recyclable": False,
        "confidence": 0.3,
        "instruction": "X",
        "instructions": ["X", "Y"],
        "extra": "bad",
    }

    with pytest.raises(ValueError, match="exactly the 5 required fields"):
        svc._validate_llm_final(raw)


def test_validate_llm_final_rejects_short_instructions() -> None:
    raw = {
        "category": "A",
        "recyclable": False,
        "confidence": 0.3,
        "instruction": "X",
        "instructions": ["only one"],
    }

    with pytest.raises(ValueError, match="at least 2"):
        svc._validate_llm_final(raw)


def test_mock_tier2_decision_handles_strict_and_uncertain() -> None:
    tier1_clean = {
        "category": "plastic",
        "confidence": 0.95,
        "top3": [],
        "escalate": False,
    }
    clean = svc._mock_tier2_decision(tier1_clean, force_cloud=True)

    assert clean["bin_type"] == "recyclable"
    assert clean["contamination_flag"] == "clean"
    assert clean.get("refined_name") == "Plastic Container"

    tier1_uncertain = {
        "category": "other_uncertain",
        "confidence": 0.4,
        "top3": [],
        "escalate": True,
    }
    uncertain = svc._mock_tier2_decision(tier1_uncertain, force_cloud=False)

    assert uncertain["bin_type"] == "non_recyclable"
    assert uncertain["needs_confirmation"] is True


def test_final_from_llm_min_normalizes_special_prefix() -> None:
    llm_final = {
        "category": "e-waste - phone",
        "recyclable": True,
        "confidence": 0.7,
        "instruction": "Use e-waste collection.",
        "instructions": ["Remove battery if safe.", "Bring to e-waste point."],
    }

    final_obj = svc._final_from_llm_min(llm_final)

    assert final_obj["category"].lower().startswith("e-waste - ")
    assert final_obj["recyclable"] is False


def test_extract_openai_error_json_handles_non_json_response() -> None:
    request = httpx.Request("GET", "https://example.com")
    response = httpx.Response(500, request=request, text="<html>bad</html>")

    data = svc._extract_openai_error_json(response)

    assert data == {}


def test_tier2_error_meta_from_http_status_error() -> None:
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx.Response(
        404,
        request=request,
        json={
            "error": {
                "type": "invalid_request_error",
                "code": "model_not_found",
                "message": "Model not found",
            }
        },
    )
    exc = httpx.HTTPStatusError("bad status", request=request, response=response)

    meta = svc._tier2_error_meta(exc)

    assert meta["http_status"] == "404"
    assert meta["code"] == "model_not_found"
    assert "Model not found" in meta["message"]


def test_tier2_error_meta_timeout() -> None:
    meta = svc._tier2_error_meta(httpx.TimeoutException("timeout"))

    assert meta["code"] == "timeout"


def test_tier2_error_meta_missing_key_runtime_error() -> None:
    exc = RuntimeError("wrapper")
    exc.__cause__ = RuntimeError("Missing OPENAI_API_KEY")

    meta = svc._tier2_error_meta(exc)

    assert meta["code"] == "missing_api_key"
