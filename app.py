from fastapi import FastAPI
import torch
import pickle
import pandas as pd

from src.models.lstm import LSTMModel
from src.models.classifier import LSTMClassifier
from src.inference.predict import ensemble_predict
from src.inference.recommend import recommend_meal

app = FastAPI()

model_reg = LSTMModel(input_size=10)
model_reg.load_state_dict(torch.load("artifacts/models/model_reg.pth"))

model_cls = LSTMClassifier(input_size=10)
model_cls.load_state_dict(torch.load("artifacts/models/model_cls.pth"))

with open("artifacts/scalers/scaler_y.pkl", "rb") as f:
    scaler_y = pickle.load(f)

df_food = pd.read_csv("data/processed/food_data.csv")


@app.post("/recommend")
def recommend(input_seq: list, goal: str = "maintain"):

    X = torch.tensor([input_seq]).float()

    pred = ensemble_predict(model_reg, model_cls, X, scaler_y)
    meals = recommend_meal(pred, df_food, goal)

    return {
        "predicted_nutrition": pred.tolist(),
        "recommended_meals": meals
    }