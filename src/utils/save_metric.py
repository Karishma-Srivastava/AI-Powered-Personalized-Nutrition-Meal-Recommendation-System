import os
import json
from datetime import datetime

def save_metrics(metrics: dict):
    os.makedirs("artifacts/metrics", exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    file_path = f"artifacts/metrics/metrics_{timestamp}.json"

    with open(file_path, "w") as f:
        json.dump(metrics, f, indent=4)