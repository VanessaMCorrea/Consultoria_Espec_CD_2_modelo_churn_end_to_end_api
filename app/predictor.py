import os
import joblib
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv()

HF_REPO_ID = os.getenv("HF_REPO_ID")
HF_FILENAME = os.getenv("HF_FILENAME")
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_REPO_ID:
    raise ValueError("HF_REPO_ID não definido no .env")

if not HF_FILENAME:
    raise ValueError("HF_FILENAME não definido no .env")

model_path = hf_hub_download(
    repo_id=HF_REPO_ID,
    filename=HF_FILENAME,
    token=HF_TOKEN
)

model = joblib.load(model_path)

def make_prediction(input_data: dict) -> dict:
    df = pd.DataFrame([input_data])

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