from typing import Union
import pickle

from fastapi import FastAPI
app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

BEST_FORECASTS = None

@app.on_event("startup")
def load_forecasts():
    global BEST_FORECASTS
    try:
        with open("forecasts1.pkl", "rb") as f:
            BEST_FORECASTS = pickle.load(f)
        print("Forecasts loaded successfully")
    except Exception as e:
        print(f"Forecast file not loaded: {e}")
        BEST_FORECASTS = {}


@app.get("/forecast")
def get_final_forecast():
    if not BEST_FORECASTS:
        return {"error": "Forecast data not available"}

    years = list(range(2025, 2031))
    response = {
        "forecasts": {},
        "calculated_total_generated_tonnes": round(
            BEST_FORECASTS.get("Total_tonnes", 0), 2
        )
    }

    for model in ["ARIMA", "WMA", "SES"]:
        response["forecasts"][model] = dict(
            zip(years, map(int, BEST_FORECASTS.get(model, [])))
        )

    return response



@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    # Fix for Alert 10021
    response.headers["X-Content-Type-Options"] = "nosniff"
    # Fix for Alert 90004
    response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
    return response
