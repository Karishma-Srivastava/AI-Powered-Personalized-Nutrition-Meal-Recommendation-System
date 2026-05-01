import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

def recommend_meal(pred_real, df, goal="maintain"):

    calories = pred_real[0]
    target = pred_real[:4]

    # ===== BASE FILTER (clean dataset) =====
    calories = pred_real[0]

    # base filter
    filtered_df = df[
        (df["Calories (kcal)"] > 50) &
        (df["Protein (g)"] > 5) &
        (df["Fat (g)"] < 20)
    ].copy()

    # goal filter
    if goal == "weight_loss":
        filtered_df = filtered_df[filtered_df["Calories (kcal)"] <= calories]
    elif goal == "weight_gain":
        filtered_df = filtered_df[filtered_df["Calories (kcal)"] >= calories]

    # ===== ADD HERE =====
    filtered_df = filtered_df[
        abs(filtered_df["Calories (kcal)"] - calories) < 100
    ]

    if len(filtered_df) < 5:
        filtered_df = df.copy()
    # ===== FEATURE MATRIX =====
    food_features = filtered_df[[
        "Calories (kcal)",
        "Protein (g)",
        "Carbohydrates (g)",
        "Fat (g)"
    ]].values

    # ===== DISTANCE =====
    distances = euclidean_distances([target], food_features)

    # keep only foods close to predicted calories
    filtered_df = filtered_df[
        abs(filtered_df["Calories (kcal)"] - calories) < 150]
    

    # ===== TOP-K SELECTION =====
    top_k_idx = distances.argsort()[0]

    selected = []

    for idx in top_k_idx:
        if idx >= len(filtered_df):
            continue

        food = filtered_df.iloc[idx]
        selected.append(food["Food_Item"])

        if len(selected) == 5:
            break
            # track keywords

            seen_types.update(food_name.split())

            if len(selected) == 5:
                break

    # ===== OPTIONAL: ADD ONE SIDE ITEM =====
    side_df = filtered_df[
        (filtered_df["Calories (kcal)"] < 150) &
        (filtered_df["Carbohydrates (g)"] > 5)
    ]
    if filtered_df.empty:
        return ["No suitable meals found"]

    filtered_df = filtered_df[
        abs(filtered_df["Calories (kcal)"] - calories) < 120
]

    if not side_df.empty:
        selected.append(side_df.sample(1)["Food_Item"].values[0])

    return selected[:5]