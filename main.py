# Record 2

import os
import pickle
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(root_path="/python")


# Fixes: "Unexpected Content-Type" for 404s
# Ensures that even errors are returned as JSON, not HTML
@app.exception_handler(404)
async def custom_404_handler(request: Request, __):
    return JSONResponse(
        status_code=404,
        content={"detail": "Not Found", "path": request.url.path},
    )

# Fixes: Security Headers (Low risk alerts in ZAP)
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    
    # Fixes: "X-Content-Type-Options Header Missing"
    response.headers["X-Content-Type-Options"] = "nosniff"
    
    # Fixes: "Insufficient Site Isolation Against Spectre Vulnerability"
    response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
    
    # Fixes: "Strict-Transport-Security Header Not Set"
    # Tells browsers to only use HTTPS
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    
    # Fixes: "Storable and Cacheable Content" (Informational)
    # Prevents sensitive API data from being cached by proxies
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    
    return response


# 1. Resolve the path immediately
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pkl_path = os.path.join(BASE_DIR, "forecast", "data", "forecasts1.pkl")

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
