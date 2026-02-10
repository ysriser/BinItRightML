from __future__ import annotations

import asyncio

import httpx

import main as root_main


async def _get(path: str):
    transport = httpx.ASGITransport(app=root_main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
        return await client.get(path)


def test_root_forecast_returns_error_when_data_missing(monkeypatch):
    monkeypatch.setattr(root_main, "BEST_FORECASTS", {})

    response = asyncio.run(_get("/forecast"))

    assert response.status_code == 200
    assert response.json() == {"error": "Forecast data not available"}


def test_root_forecast_returns_expected_payload(monkeypatch):
    monkeypatch.setattr(
        root_main,
        "BEST_FORECASTS",
        {
            "Total_tonnes": 99.9,
            "ARIMA": [1, 2, 3, 4, 5, 6],
            "WMA": [6, 5, 4, 3, 2, 1],
            "SES": [9, 9, 9, 9, 9, 9],
        },
    )

    response = asyncio.run(_get("/forecast"))
    payload = response.json()

    assert response.status_code == 200
    assert payload["calculated_total_generated_tonnes"] == 99.9
    assert payload["forecasts"]["ARIMA"]["2025"] == 1


def test_root_404_returns_json_payload():
    response = asyncio.run(_get("/not-found"))

    assert response.status_code == 404
    body = response.json()
    assert body["detail"] == "Not Found"
    assert body["path"] == "/python/not-found"


def test_root_forecast_adds_security_headers(monkeypatch):
    monkeypatch.setattr(root_main, "BEST_FORECASTS", {})

    response = asyncio.run(_get("/forecast"))

    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["Cross-Origin-Resource-Policy"] == "same-origin"
    assert "max-age=31536000" in response.headers["Strict-Transport-Security"]
    assert response.headers["Cache-Control"] == "no-store, no-cache, must-revalidate"
