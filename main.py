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

with open("forecasts1.pkl", "rb") as f:
    BEST_FORECASTS = pickle.load(f)

@app.get("/forecast")
def get_final_forecast():

    years = list(range(2025, 2031))

    response = {
        "forecasts": {},
        "calculated_total_generated_tonnes": round(
            BEST_FORECASTS["Total_tonnes"], 2
        )
    }

    for model_name in ["ARIMA", "WMA", "SES"]:
        response["forecasts"][model_name] = dict(
            zip(years, map(int, BEST_FORECASTS[model_name]))
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
