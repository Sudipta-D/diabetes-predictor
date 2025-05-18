# Diabetes Predictor

👋 Welcome! This sweet little app predicts the probability of diabetes (as a percentage) based on your health measurements and tells you whether you're likely diabetic or not.

## 💡 What It Does

* Collects your health data (e.g., age, glucose level, BMI)
* Utilizes a trained XGBoost model to calculate diabetes probability
* Shows a friendly message (healthy, at risk, or diabetic) and displays your exact risk percentage

## 🔍 Risk Categories

Based on the predicted probability:

* **< 50%**: ✅ You appear healthy
* **50% – 74%**: 🔶 You are at risk of diabetes
* **≥ 75%**: ⚠️ You are diabetic

## 🛠️ Get Started

1. **Clone** this repository:

   ```bash
   git clone https://github.com/Sudipta-D/diabetes-predictor.git
   cd diabetes-predictor
   ```

2. **Install** the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run** the Streamlit app:

   ```bash
   streamlit run app.py
   ```

4. **Visit** the app in your browser:

   > [http://localhost:8501](http://localhost:8501)

Enjoy exploring your risk assessment!

## 📝 Explore Behind the Scenes

Peek into `notebooks/diabetes_pipeline.ipynb` to see data cleaning, SMOTE balancing, model training, and insights.

## ❤️ Thank You

Feel free to ⭐ the repo, open issues, or suggest improvements. Happy exploring and stay healthy!

— Sudipta D


