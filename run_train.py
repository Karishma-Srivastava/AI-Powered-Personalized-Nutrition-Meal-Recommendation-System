print("🚀 Training started")

# ===== LOAD DATA =====
from src.data.load_data import load_data
df = load_data("data/raw/daily_food_nutrition_dataset.csv")

# ===== PREPROCESS =====
from src.data.preprocess import preprocess
X, y_reg, y_cls = preprocess(df)

# ===== SCALE =====
from sklearn.preprocessing import StandardScaler
import pickle, os

scaler_X = StandardScaler()
X_reshaped = X.reshape(-1, X.shape[-1])
X_scaled = scaler_X.fit_transform(X_reshaped)
X = X_scaled.reshape(X.shape)

scaler_y = StandardScaler()
y_reg = scaler_y.fit_transform(y_reg)

# ===== SAVE SCALERS =====
os.makedirs("artifacts/scalers", exist_ok=True)

with open("artifacts/scalers/scaler_X.pkl", "wb") as f:
    pickle.dump(scaler_X, f)

with open("artifacts/scalers/scaler_y.pkl", "wb") as f:
    pickle.dump(scaler_y, f)

# ===== SPLIT =====
from sklearn.model_selection import train_test_split

X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
    X, y_reg, y_cls, test_size=0.2, random_state=42
)

# ===== TRAIN =====
from src.training.train import train_model

model_reg, model_cls = train_model(
    X_train, y_reg_train, y_cls_train,
    X_test, y_reg_test, y_cls_test,
    scaler_y
)

# ===== SAVE MODELS =====
import torch

os.makedirs("artifacts/models", exist_ok=True)

torch.save(model_reg.state_dict(), "artifacts/models/model_reg.pth")
torch.save(model_cls.state_dict(), "artifacts/models/model_cls.pth")

print("💾 Models saved")

# ===== METRICS =====
from src.utils.save_metric import save_metrics
save_metrics({"status": "training_completed"})

print("🎯 Training completed")