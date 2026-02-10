import asyncio

import httpx

import main


async def _get_forecast():
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
        return await client.get("/forecast")


def test_forecast_returns_error_when_data_missing(monkeypatch):
    monkeypatch.setattr(main, "BEST_FORECASTS", {})

    response = asyncio.run(_get_forecast())

    assert response.status_code == 200
    assert response.json() == {"error": "Forecast data not available"}


def test_forecast_returns_expected_shape(monkeypatch):
    monkeypatch.setattr(
        main,
        "BEST_FORECASTS",
        {
            "Total_tonnes": 123.456,
            "ARIMA": [100, 101, 102, 103, 104, 105],
            "WMA": [200, 201, 202, 203, 204, 205],
            "SES": [300, 301, 302, 303, 304, 305],
        },
    )

    response = asyncio.run(_get_forecast())
    payload = response.json()

    assert response.status_code == 200
    assert payload["calculated_total_generated_tonnes"] == 123.46
    assert set(payload["forecasts"].keys()) == {"ARIMA", "WMA", "SES"}
    assert payload["forecasts"]["ARIMA"]["2025"] == 100
    assert payload["forecasts"]["SES"]["2030"] == 305
