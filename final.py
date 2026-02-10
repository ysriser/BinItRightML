from fastapi import FastAPI
import pickle

app = FastAPI()

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

