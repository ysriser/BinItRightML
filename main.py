import os
import pickle
from fastapi import FastAPI
from typing import Union

app = FastAPI(root_path="/python")

# 1. Resolve the path immediately
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pkl_path = os.path.join(BASE_DIR, "forecasts1.pkl")

# 2. Load the data at the TOP LEVEL (No more startup event needed)
print(f"--- Attempting to load: {pkl_path} ---")
try:
    with open(pkl_path, "rb") as f:
        BEST_FORECASTS = pickle.load(f)
    print("--- SUCCESS: Forecasts loaded ---")
except Exception as e:
    print(f"--- ERROR: {e} ---")
    BEST_FORECASTS = {}

# 3. Your endpoints remain the same
@app.get("/forecast")
def get_final_forecast():
    # If the dictionary is empty, it returns the error you saw in curl
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