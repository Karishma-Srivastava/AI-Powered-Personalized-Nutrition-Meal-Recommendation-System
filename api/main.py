from fastapi import FastAPI
import torch
import numpy as np
import pickle
import pandas as pd

from src.models.lstm import LSTMModel
from src.models.classifier import LSTMClassifier
from src.inference.predict import ensemble_predict
from src.inference.recommend import recommend_meal

app = FastAPI()

# =========================
# LOAD MODELS
# =========================
model_reg = LSTMModel(input_size=10)
model_reg.load_state_dict(torch.load("artifacts/models/model_reg.pth"))

model_cls = LSTMClassifier(input_size=10)
model_cls.load_state_dict(torch.load("artifacts/models/model_cls.pth"))

model_reg.eval()
model_cls.eval()

# =========================
# LOAD SCALERS
# =========================
with open("artifacts/scalers/scaler_X.pkl", "rb") as f:
    scaler_X = pickle.load(f)

with open("artifacts/scalers/scaler_y.pkl", "rb") as f:
    scaler_y = pickle.load(f)

# =========================
# LOAD FOOD DATA
# =========================
df_food = pd.read_csv("data/processed/food_data.csv")

# =========================
# API ENDPOINT
# =========================
@app.post("/predict")
def predict(data: dict):
    """
    Input:
    {
        "sequence": [[7 days × 10 features]],
        "goal": "weight_loss" | "maintain" | "weight_gain"
    }
    """

    seq = np.array(data["sequence"])

    # ===== SCALE INPUT =====
    seq_reshaped = seq.reshape(-1, seq.shape[-1])
    seq_scaled = scaler_X.transform(seq_reshaped)
    seq_scaled = seq_scaled.reshape(seq.shape)

    X = torch.tensor(seq_scaled).float()

    # ===== ENSEMBLE PREDICTION =====
    pred_real = ensemble_predict(model_reg, model_cls, X, scaler_y)

    # ===== RECOMMENDATION =====
    goal = data.get("goal", "maintain")
    meals = recommend_meal(pred_real, df_food, goal)

    return {
        "predicted_nutrition": pred_real.tolist(),
        "recommended_meals": meals
    }