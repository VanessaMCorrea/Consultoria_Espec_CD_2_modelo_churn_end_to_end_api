import os
import joblib
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv()

HF_REPO_ID = os.getenv("HF_REPO_ID", "VanessaMCorrea/churn-telecom-model")
HF_FILENAME = os.getenv("HF_FILENAME", "churn_pipeline.joblib")

model_path = hf_hub_download(
    repo_id=HF_REPO_ID,
    filename=HF_FILENAME
)

model = joblib.load(model_path)

def make_prediction(input_data: dict) -> dict:
    # ✅ NOVO: tratamento de erro na predição
    try:
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

    except ValueError as e:
        raise ValueError(f"Erro nos dados de entrada: {e}")
    except Exception as e:
        raise RuntimeError(f"Erro inesperado na predição: {e}")