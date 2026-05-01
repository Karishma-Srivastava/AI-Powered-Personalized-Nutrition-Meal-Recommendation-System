import pandas as pd
import os

df = pd.read_csv(
    "data/raw/daily_food_nutrition_dataset.csv",
    on_bad_lines="skip"
)

# ensure column exists
if "Food" in df.columns:
    df = df.rename(columns={"Food": "Food_Item"})

os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/food_data.csv", index=False)

print("✅ food_data.csv created")