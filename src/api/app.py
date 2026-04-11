from fastapi import FastAPI, Body
import pandas as pd
from src.xgboost.inference.infer import run_inference

app = FastAPI()

@app.get("/")
def home():
    return {"message": "SLA Anomaly Detection API running"}

@app.post("/predict")
def predict(data: list = Body(...)):
    df = pd.DataFrame(data)
    results = run_inference(df)
    return results.to_dict(orient="records")