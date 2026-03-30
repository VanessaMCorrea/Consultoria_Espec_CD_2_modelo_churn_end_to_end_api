import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "churn_pipeline.joblib")

model = joblib.load(MODEL_PATH)

def make_prediction(input_data: dict) -> dict:
    df = pd.DataFrame([input_data])

    # garantir coerência com o treino
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    pred = int(model.predict(df)[0])

    result = {
        "churn_predito": pred,
        "churn_label": "Yes" if pred == 1 else "No"
    }

    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(df)[0][1])
        result["probabilidade_churn"] = round(proba, 4)

    return result