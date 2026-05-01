import numpy as np

def get_calorie_class(cal):
    if cal < 200:
        return 0
    elif cal < 400:
        return 1
    else:
        return 2


def preprocess(df, seq_len=7):

    # ===== COPY (avoid mutation) =====
    df = df.copy()

    # ===== OPTIONAL: ensure time order =====
    df = df.reset_index(drop=True)

    # ===== BASE FEATURES =====
    base_features = [
        "Calories (kcal)",
        "Protein (g)",
        "Carbohydrates (g)",
        "Fat (g)"
    ]

    # ===== FEATURE ENGINEERING =====
    df["calorie_diff"] = df["Calories (kcal)"].diff().fillna(0)
    df["protein_diff"] = df["Protein (g)"].diff().fillna(0)

    df["rolling_calorie"] = df["Calories (kcal)"].rolling(3).mean().ffill().fillna(0)
    df["rolling_protein"] = df["Protein (g)"].rolling(3).mean().ffill().fillna(0)

    df["day_index"] = np.arange(len(df)) % 7
    df["is_weekend"] = (df["day_index"] >= 5).astype(int)

    features = base_features + [
        "calorie_diff",
        "protein_diff",
        "rolling_calorie",
        "rolling_protein",
        "day_index",
        "is_weekend"
    ]

    # ===== CLEAN =====
    df = df[features].dropna()

    values = df.values

    # ===== TARGET =====
    target_cols = base_features
    target_values = df[target_cols].values

    X, y_reg, y_cls = [], [], []

    for i in range(len(values) - seq_len):
        X.append(values[i:i+seq_len])
        y_reg.append(target_values[i+seq_len])

        cal = target_values[i+seq_len][0]
        y_cls.append(get_calorie_class(cal))

    return np.array(X), np.array(y_reg), np.array(y_cls)