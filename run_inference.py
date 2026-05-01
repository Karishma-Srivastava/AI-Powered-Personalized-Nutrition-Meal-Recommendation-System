import torch
import pickle
import pandas as pd

from src.models.transformer import TransformerModel
from src.inference.predict import predict
from src.inference.recommend import recommend_meal
from src.data.load_data import load_data
from src.data.preprocess import preprocess

# =========================
# LOAD DATA
# =========================
df = load_data("data/raw/daily_food_nutrition_dataset.csv")
X, y = preprocess(df)

X_test = torch.tensor(X).float()

# =========================
# LOAD MODEL
# =========================
model = TransformerModel(input_size=4)
model.load_state_dict(torch.load("artifacts/models/model.pth"))
model.eval()

# =========================
# LOAD SCALER
# =========================
with open("artifacts/scalers/scaler_X.pkl", "rb") as f:
    scaler_X = pickle.load(f)

with open("artifacts/scalers/scaler_y.pkl", "rb") as f:
    scaler_y = pickle.load(f)

# =========================
# RUN PREDICTION
# =========================
pred_scaled = model(X_test)[0].detach().numpy()
pred = scaler_y.inverse_transform([pred_scaled])[0]


# LOAD FOOD DATA

df_food = pd.read_csv("data/processed/food_data.csv")

meal = recommend_meal(pred, df_food)

print("Predicted nutrition:", pred)
print("Recommended meal:", meal)