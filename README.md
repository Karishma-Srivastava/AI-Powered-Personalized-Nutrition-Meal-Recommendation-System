# 🍽️ Personalized Nutrition & Meal Recommendation System

## 📌 Overview

This project is an end-to-end machine learning system that predicts a user's next-day nutritional needs based on historical eating patterns and generates realistic meal recommendations.

It combines:

* Time-series modeling (Transformer)
* Feature engineering
* Constraint-based recommendation system

---

## 🚀 Key Features

* 📊 **Time-Series Prediction** using Transformer (PyTorch)
* 🧠 **Behavior Modeling** with rolling trends and nutrition ratios
* 🍽️ **Smart Meal Recommendation Engine**
* ⚖️ **Constraint-based Optimization** for realistic meals
* 🧩 Multi-item meal generation (main + side)

---

## 🏗️ Pipeline

Raw Data → Feature Engineering → Sequence Builder (7-day)
→ Transformer Model → Nutrition Prediction
→ Food Filtering → Meal Construction → Final Recommendation

---

## 📂 Project Structure

fitness-recommender/
├── data/
├── artifacts/
├── src/
│   ├── models/
│   ├── pipeline/
│   ├── inference/
│   └── utils/
├── api/
├── run_train.py
├── run_inference.py

---

## 🧠 Model Details

* Model: Transformer Encoder
* Input: 7-day nutrition sequence
* Output: Next-day nutrition vector
* Loss: MSE
* Features:

  * Macronutrients
  * Ratios (Protein/Calories)
  * Rolling averages
  * Trends

---

## 🍽️ Recommendation System

Instead of direct food prediction, the system:

1. Predicts nutritional target
2. Selects best matching main dish
3. Adds optimized side dish
4. Applies constraints for realism

---

## 📊 Results

* Stable convergence during training
* No overfitting observed
* Realistic multi-item meal recommendations generated

---

## 💡 Key Learnings

* Importance of feature engineering in time-series data
* Difference between mathematical optimization vs real-world usability
* Role of constraints in recommendation systems

---

## 🔮 Future Work

* Full-day meal planning (breakfast/lunch/dinner)
* Goal-based diet (weight loss / muscle gain)
* Real-world data integration (FitRec)

---

## ⚙️ Tech Stack

* Python
* PyTorch
* Pandas, NumPy
* Scikit-learn

---

## 👩‍💻 Author

Karishma Srivastava
