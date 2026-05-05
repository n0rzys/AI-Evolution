import json
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import torch
import torch.nn as nn
from pathlib import Path
from pydantic import BaseModel

class HeartFeatures(BaseModel):
    BMI: float
    Smoking: str
    AlcoholDrinking: str
    Stroke: str
    PhysicalHealth: float
    MentalHealth: float
    DiffWalking: str
    Sex: str
    AgeCategory: str
    Race: str
    Diabetic: str
    PhysicalActivity: str
    GenHealth: str
    SleepTime: float
    Asthma: str
    KidneyDisease: str
    SkinCancer: str

BASE_DIR = Path(__file__).resolve().parent.parent
ML_DIR = BASE_DIR / "ml"

age_order = [
    '18-24', '25-29', '30-34', '35-39', '40-44', '45-49',
    '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80 or older'
]
age_map = {val: i for i, val in enumerate(age_order)}

gen_health_map = {
    'Poor': 0,
    'Fair': 1,
    'Good': 2,
    'Very good': 3,
    'Excellent': 4
}

numeric_col = ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime', 'AgeCategory', 'GenHealth']


class MLP(nn.Module):
    def __init__(self, in_dim, num_classes, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class ModelService:
    def __init__(self):
        with open(ML_DIR / "encoder.pkl", "rb") as f:
            self.encoder = pickle.load(f)

        with open(ML_DIR / "scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)

        self.xgb_model = xgb.XGBClassifier()
        self.xgb_model.load_model(ML_DIR / "model_xgb.json")

        with open(ML_DIR / "mlp_meta.json", "r", encoding="utf-8") as f:
            mlp_meta = json.load(f)

        self.mlp_model = MLP(
            in_dim=mlp_meta["input_dim"],
            num_classes=mlp_meta["num_classes"]
        )
        self.mlp_model.load_state_dict(
            torch.load(ML_DIR / "model_mlp.pth", map_location="cpu")
        )
        self.mlp_model.eval()

    def preprocess(self, features: dict):
        df = pd.DataFrame([features])

        df["AgeCategory"] = df["AgeCategory"].map(age_map)
        df["GenHealth"] = df["GenHealth"].map(gen_health_map)

        categorical_col = [c for c in df.columns if c not in numeric_col]

        x_cat = self.encoder.transform(df[categorical_col])
        x_num = self.scaler.transform(df[numeric_col])

        x = np.hstack([x_cat, x_num])
        return x

    def predict_xgb(self, features: dict):
        x = self.preprocess(features)
        pred = self.xgb_model.predict(x)[0]
        prob = self.xgb_model.predict_proba(x)[0][1]
        return {
            "model": "xgboost",
            "prediction": int(pred),
            "probability": float(prob)
        }

    def predict_mlp(self, features: dict):
        x = self.preprocess(features)
        x_tensor = torch.tensor(x, dtype=torch.float32)

        with torch.no_grad():
            logits = self.mlp_model(x_tensor)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            prob = probs[0][1].item()

        return {
            "model": "mlp",
            "prediction": int(pred),
            "probability": float(prob)
        }