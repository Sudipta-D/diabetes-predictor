import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import base64
import plotly.graph_objects as go

# Load Model
model_path = os.path.join('model', 'best_xgb_model.pkl')
model = pickle.load(open(model_path, 'rb'))

# ----- Styling -----
st.set_page_config(layout="centered", page_title="Diabetes Predictor", page_icon="🧬")

# Background Image
def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_background("images/bg.png")
# Sidebar
page = st.sidebar.radio("Go to", ["🔍 Prediction", "📁 Project Info", "ℹ️ About"])

# Risk Gauge Chart
def show_risk_gauge(prob_percent):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_percent,
        title={'text': "Diabetes Risk (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#E74C3C" if prob_percent >= 75 else "#F1C40F" if prob_percent >= 50 else "#2ECC71"},
            'steps': [
                {'range': [0, 50], 'color': "#dff0d8"},
                {'range': [50, 75], 'color': "#fcf8e3"},
                {'range': [75, 100], 'color': "#f2dede"},
            ],
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

# Main App Pages
if page == "🔍 Prediction":
    st.title("🧪 Diabetes Risk Prediction")
    st.markdown("Estimate your diabetes risk using 21 lifestyle and health indicators.")
    st.markdown("---")

    with st.form("21_feature_form"):
        # ─── Demographics ─────────────────
        st.subheader("👤 Demographics")
        age = st.slider("Age (years)", 18, 100, 30)
        sex = st.selectbox("Sex", ["Female", "Male"])
        sex_val = 1 if sex == "Male" else 0
        education = st.slider("Education (0=None to 6+)", 0, 6, 2)
        income = st.slider("Income Level (0=Low to 10=High)", 0, 10, 5)

        # ─── Health & Access ─────────────────
        st.subheader("🩺 Health & Access")
        bmi = st.slider("BMI", 10.0, 60.0, 25.0)
        gen_hlth = st.slider("General Health (1=Excellent → 5=Poor)", 1, 5, 3)
        phys_hlth = st.slider("Physical Health Issues (last 30 days)", 0, 30, 0)
        ment_hlth = st.slider("Mental Health Issues (last 30 days)", 0, 30, 0)
        any_health = st.radio("Do you have Health Coverage?", ["Yes", "No"])
        no_doc = st.radio("Avoided Doctor Due to Cost?", ["No", "Yes"])
        any_health = 1 if any_health == "Yes" else 0
        no_doc = 1 if no_doc == "Yes" else 0

        # ─── Lifestyle & Medical ─────────────────
        st.subheader("🏃 Lifestyle & Medical Conditions")
        high_bp = st.checkbox("High Blood Pressure")
        high_chol = st.checkbox("High Cholesterol")
        chol_check = st.checkbox("Had Cholesterol Check")
        smoker = st.checkbox("Smoker")
        stroke = st.checkbox("History of Stroke")
        heart_disease = st.checkbox("Heart Disease")
        phys_act = st.checkbox("Physically Active")
        fruits = st.checkbox("Eat Fruits Daily")
        veggies = st.checkbox("Eat Vegetables Daily")
        heavy_alc = st.checkbox("Heavy Alcohol Use")
        diff_walk = st.checkbox("Difficulty Walking")

        # ─── Submit ─────────────────
        submitted = st.form_submit_button("🔍 Predict")

    if submitted:
        features = np.array([[
            int(high_bp), int(high_chol), int(chol_check), bmi, int(smoker),
            int(stroke), int(heart_disease), int(phys_act), int(fruits), int(veggies),
            int(heavy_alc), int(any_health), int(no_doc), gen_hlth, ment_hlth,
            phys_hlth, int(diff_walk), sex_val, age, education, income
        ]])

        proba = model.predict_proba(features)[0][1]
        proba_percent = round(proba * 100, 2)

        if proba >= 0.75:
            st.error("⚠️ You are diabetic")
        elif proba >= 0.5:
            st.warning("🔶 You are at risk of diabetes")
        else:
            st.success("✅ You appear healthy")

        st.markdown(f"### 🔢 Predicted Risk: `{proba_percent:.1f}%`")
        show_risk_gauge(proba_percent)



# Project Info
elif page == "📁 Project Info":
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.header("📊 Project Overview")
    st.write("""
    This project aims to predict diabetes onset using health and lifestyle data from the BRFSS 2015 dataset. It involves data cleaning, exploratory analysis, model building (Logistic Regression, Random Forest, XGBoost), hyperparameter tuning, and interpretability with SHAP to deliver an accurate and explainable prediction model.
    """)

    st.markdown("""
    ### 🔧 Tech Stack:

    - **Language & Environment:** Python  
    - **Data Handling & Analysis:** pandas, NumPy  
    - **Visualization:** matplotlib, seaborn, SHAP  
    - **Modeling:** scikit-learn, XGBoost  
    - **Hyperparameter Tuning:** GridSearchCV, RandomizedSearchCV  
    - **Model Interpretability:** SHAP  
    - **App Deployment:** Streamlit  

    **🎯 Target Variable:** Presence of Diabetes  
    **📂 Source:** Publicly available health dataset - BRFSS 2015 (balanced 50-50)  
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# About Page
elif page == "ℹ️ About":
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.header("👤 About the Developer")
    st.write("""
    Created with ❤️ by **Sudipta Das**, a data analyst in the making, blending statistics, storytelling, and software to build human-centric solutions.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

