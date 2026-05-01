import pandas as pd

def load_data(file_path):
    print("📂 Loading data from:", file_path)
    return pd.read_csv(file_path, on_bad_lines='skip')


# def load_data(path):
#     import pandas as pd
#     return pd.read_csv(path)