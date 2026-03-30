---
language: pt
tags:
  - sklearn
  - classification
  - customer-churn
  - telecom
  - mlops
---

# Churn Predictor - Telecom Customers v1

Modelo de classificação binária desenvolvido para prever a probabilidade de **churn** (evasão) de clientes de uma empresa de telecomunicações.

Este projeto foi desenvolvido na disciplina **Consultoria Especializada em Ciência de Dados 2 (PUC-SP)** e contempla um fluxo completo de MLOps, incluindo:

- treinamento do modelo
- criação de pipeline com pré-processamento
- serialização com joblib
- publicação no Hugging Face
- consumo via API FastAPI

---

## Uso Rápido

Para carregar o modelo diretamente do Hugging Face e realizar uma predição em Python:

```python
from huggingface_hub import hf_hub_download
import joblib
import pandas as pd

# Download do modelo
repo_id = "VanessaMCorrea/churn-telecom-model"
model_path = hf_hub_download(repo_id=repo_id, filename="churn_pipeline.joblib")

# Carregar modelo
model = joblib.load(model_path)

# Exemplo de entrada
data = pd.DataFrame([{
    "gender": "Female",
    "SeniorCitizen": "0",
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 89.5,
    "TotalCharges": 1074.0
}])

# Predição
prediction = model.predict(data)[0]

print(f"Resultado: {'Churn' if prediction == 1 else 'Ativo'}")