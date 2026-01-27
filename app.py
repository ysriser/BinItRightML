
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle


app = FastAPI()

with open("final_forecasts.pkl", "rb") as f:
    BEST_FORECASTS = pickle.load(f)

@app.get("/forecast/final")
def get_final_forecast():

    years = list(range(2025, 2031))

    response = {}

    for target, info in BEST_FORECASTS.items():
        response[target] = {
            "model": info["best_model"],
            "forecast": dict(zip(years, map(int, info["forecast"])))
        }

    return response
