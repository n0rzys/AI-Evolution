from fastapi import FastAPI
from api.model_loader import ModelService, HeartFeatures

app = FastAPI(
    title="AI Evolution Project API",
    description="Сервис для сравнения современных ML-моделей в задаче предсказания сердечных заболеваний",
    version="1.0.0"
)

model_service = ModelService()

@app.get("/")
def root():
    return {
        "message": "API работает",
        "available_endpoints": [
            "/predict/xgb",
            "/predict/mlp"
        ]
    }


@app.post("/predict/xgb")
def predict_xgb(features: HeartFeatures):
    result = model_service.predict_xgb(features.model_dump())
    return result


@app.post("/predict/mlp")
def predict_mlp(features: HeartFeatures):
    result = model_service.predict_mlp(features.model_dump())
    return result