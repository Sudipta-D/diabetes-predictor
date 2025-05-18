# 🩺 Diabetes Predictor App

A machine learning-based Streamlit web application that predicts the likelihood of diabetes in individuals using health and lifestyle data.

## 📌 About the Project

This project leverages the balanced BRFSS 2015 dataset to train and evaluate predictive models for early diabetes detection. The system returns a probability score and classification indicating whether a person is likely to have diabetes. It supports informed health decisions with interpretable SHAP explanations.

### 🔧 Tech Stack

- **Language & Environment:** Python  
- **Data Handling & Analysis:** pandas, NumPy  
- **Visualization:** matplotlib, seaborn, SHAP  
- **Modeling:** scikit-learn, XGBoost  
- **Hyperparameter Tuning:** GridSearchCV, RandomizedSearchCV  
- **Model Interpretability:** SHAP  
- **Deployment:** Streamlit  

### 🎯 Target Variable

- `Diabetes_binary`: Indicates whether the person has diabetes or not.

  💡 What It Does

Collects your health data (e.g., age, glucose level, BMI)

Utilizes a trained XGBoost model to calculate diabetes probability

Shows a friendly message (healthy, at risk, or diabetic) and displays your exact risk percentage

🔍 Risk Categories

Based on the predicted probability:

< 50%: ✅ You appear healthy

50% – 74%: 🔶 You are at risk of diabetes

≥ 75%: ⚠️ You are diabetic

### 📁 Dataset Source

- Publicly available health dataset: [BRFSS 2015](https://www.cdc.gov/brfss/index.html) (pre-balanced 50-50 for model performance)

### 🚀 App Deployment

- 🔗 [Live Streamlit App](https://sudipta-d-diabetes-predictor.streamlit.app/)

---

## 📊 Models Used

- Logistic Regression  
- Random Forest  
- XGBoost (Tuned – Final Model)

## 📈 Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1-Score  
- ROC-AUC  
- SHAP-based model interpretability

---

## 📂 Folder Structure

```
├── app.py                     # Streamlit app
├── data/
│   └── raw.csv                # Dataset used
├── images/
│   └── bg.png                 # UI background image
├── model/
│   └── best_xgb_model.pkl     # Tuned XGBoost model
├── notebooks/
│   └── diabetes_pipeline.ipynb # Full EDA + training notebook
├── utils/
│   └── preprocessing.py       # Custom preprocessing functions
├── requirements.txt
└── README.md
```

---

## ✅ Future Enhancements

- Add more visual feedback for predictions
- Dockerize the app
- Add user input validation & error handling


