# 🍽️ AI-Powered Personalized Nutrition & Meal Recommendation System (LSTM + Ensemble + FastAPI)

🚀 Overview

This project builds an end-to-end machine learning system that predicts daily nutritional requirements from past intake and recommends meals accordingly.

It combines:

Time-series modeling (LSTM)
Classification-based ensemble correction
Rule-based filtering for practical meal recommendations
FastAPI deployment for real-time inference

⚙️ Architecture

Input (7-day sequence)
        ↓
Feature Engineering (trend, rolling stats, temporal signals)
        ↓
LSTM Regression Model → Predict nutrition (Calories, Protein, Carbs, Fat)
        ↓
LSTM Classification Model → Low / Medium / High calorie class
        ↓
Ensemble Adjustment
        ↓
Rule-based Filtering + Similarity Matching
        ↓
Meal Recommendation
        ↓
FastAPI API

🧪 Model Details

Regression Model

LSTM (input_size = 10, hidden = 64)
Output: 4 nutritional values
Loss: MSE

Classification Model

LSTM-based classifier
Output: 3 classes (low / medium / high calorie)
Loss: CrossEntropy
Ensemble Strategy

Classification adjusts regression output:
Low → ×0.8
Medium → ×1.0
High → ×1.2

📊 Evaluation

Baseline:

Predict next day = last day

Model:

~50% reduction in MSE compared to baseline

⚠️ Limitations
Model tends to regress toward mean (underestimates spikes)
No personalization (user profile missing)
Limited contextual features (activity, goals, preferences)

🚀 API Usage

Run server
uvicorn api.main:app --reload
Endpoint
POST /predict
Sample Request
{
  "sequence": [[[...7 days × 10 features...]]],
  "goal": "maintain"
}
Sample Response
{
  "predicted_nutrition": [...],
  "recommended_meals": [...],
  "reason": "Meals selected based on predicted calorie target"
}

--  Screenshot:

<img width="1137" height="686" alt="image" src="https://github.com/user-attachments/assets/07085586-2135-4349-afbe-07fc6ac519a1" />

📦 Tech Stack

Python
PyTorch
Scikit-learn
Pandas
FastAPI


fitness-recommender/
│
├── api/
│   └── main.py
│
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── inference/
│
├── artifacts/
│   ├── models/
│   ├── scalers/
│
├── data/
│
├── run_train.py
├── run_inference.py
├── README.md
└── requirements.txt
