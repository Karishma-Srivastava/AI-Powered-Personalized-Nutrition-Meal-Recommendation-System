def build_sequences(df, features, seq_len=7):
    import numpy as np

    X, y = [], []

    for user in df["user_id"].unique():
        user_df = df[df["user_id"] == user]

        vals = user_df[features].values

        for i in range(len(vals) - seq_len):
            X.append(vals[i:i+seq_len])
            y.append(vals[i+seq_len])

    return np.array(X), np.array(y)