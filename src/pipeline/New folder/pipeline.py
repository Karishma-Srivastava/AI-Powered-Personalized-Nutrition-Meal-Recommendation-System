from src.data.load_data import load_nutrition_data
from src.data.preprocess import preprocess
from src.data.feature_engineering import add_features

def run_pipeline(path):
    df = load_nutrition_data(path)
    df = preprocess(df)
    df = add_features(df)
    return df