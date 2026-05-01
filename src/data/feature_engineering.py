def add_features(df):
    df["Protein_Ratio"] = df["Protein"] / (df["Calories"] + 1)
    return df