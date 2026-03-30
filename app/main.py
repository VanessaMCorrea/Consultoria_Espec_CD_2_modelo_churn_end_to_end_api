from fastapi import FastAPI
from app.schemas import ChurnInput
from app.predictor import make_prediction

app = FastAPI(
    title="Churn Prediction API",
    description="API para previsão de churn de clientes de telecomunicações",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"message": "API de churn funcionando"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: ChurnInput):
    return make_prediction(payload.model_dump())