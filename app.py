# ================================================================
#  🧠 Stroke Prediction — Streamlit Web App
#  Final Project Deployment
#  Dataset: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
# ================================================================
import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Stroke Risk Predictor",
    page_icon="🧠",
    layout="centered"
)

# ── Load saved artifacts ─────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model    = joblib.load("saved_model.pkl")
    scaler   = joblib.load("saved_scaler.pkl")
    encoders = joblib.load("saved_encoders.pkl")
    features = joblib.load("saved_features.pkl")
    return model, scaler, encoders, features

model, scaler, encoders, top_features = load_artifacts()

# ── Header ───────────────────────────────────────────────────────
st.title("🧠 Stroke Risk Prediction App")
st.markdown("""
Provide the patient's clinical and demographic details.  
The model will estimate the **probability of stroke**.

> **Model:** Tuned Random Forest Classifier (GridSearchCV)  
> **Dataset:** Kaggle Stroke Prediction Dataset — 5,110 patients  
> **Validation:** Stratified 10-Fold Cross-Validation
""")
st.divider()

# ── Patient inputs ────────────────────────────────────────────────
st.sidebar.header("🩺 Patient Information")

gender         = st.sidebar.selectbox("Gender", ["Male", "Female"])
age            = st.sidebar.slider("Age", 0, 82, 45)
hypertension   = st.sidebar.selectbox("Hypertension", [0, 1],
                    format_func=lambda x: "Yes (1)" if x else "No (0)")
heart_disease  = st.sidebar.selectbox("Heart Disease", [0, 1],
                    format_func=lambda x: "Yes (1)" if x else "No (0)")
ever_married   = st.sidebar.selectbox("Ever Married", ["Yes", "No"])
work_type      = st.sidebar.selectbox("Work Type",
                    ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
residence      = st.sidebar.selectbox("Residence Type", ["Urban", "Rural"])
avg_glucose    = st.sidebar.slider("Avg Glucose Level (mg/dL)", 55.0, 272.0, 100.0, step=0.5)
bmi            = st.sidebar.slider("BMI", 10.0, 98.0, 28.0, step=0.1)
smoking        = st.sidebar.selectbox("Smoking Status",
                    ["never smoked", "formerly smoked", "smokes", "Unknown"])

# ── Build input dict ──────────────────────────────────────────────
glucose_bmi_ratio = avg_glucose / bmi

raw_input = {
    "gender"           : gender,
    "age"              : age,
    "hypertension"     : hypertension,
    "heart_disease"    : heart_disease,
    "ever_married"     : ever_married,
    "work_type"        : work_type,
    "Residence_type"   : residence,
    "avg_glucose_level": avg_glucose,
    "bmi"              : bmi,
    "smoking_status"   : smoking,
    "glucose_bmi_ratio": glucose_bmi_ratio,
}

# ── Patient summary ───────────────────────────────────────────────
st.subheader("📋 Patient Summary")
summary = pd.DataFrame([raw_input]).T.rename(columns={0: "Value"})
st.dataframe(summary, use_container_width=True)

# ── Encode ────────────────────────────────────────────────────────
input_df = pd.DataFrame([raw_input])
cat_cols  = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]

for col in cat_cols:
    le  = encoders[col]
    val = input_df[col].values[0]
    input_df[col] = le.transform([val])[0] if val in le.classes_ else 0

# Select only the features the model was trained on
input_selected = input_df[top_features]

# ── Predict ───────────────────────────────────────────────────────
st.divider()
if st.button("🔍 Predict Stroke Risk", use_container_width=True, type="primary"):
    prediction  = model.predict(input_selected)[0]
    probability = model.predict_proba(input_selected)[0][1]

    st.subheader("🎯 Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ **HIGH Stroke Risk Detected**\n\nProbability: **{probability:.1%}**")
        st.markdown("""
        > The model predicts this patient is **at high risk of stroke**.  
        > Immediate consultation with a neurologist is strongly recommended.
        """)
    else:
        st.success(f"✅ **LOW Stroke Risk**\n\nProbability: **{probability:.1%}**")
        st.markdown("""
        > The model predicts this patient has a **low risk of stroke**.  
        > Continue healthy lifestyle habits and regular check-ups.
        """)

    st.progress(float(probability), text=f"Stroke Risk Score: {probability:.1%}")

    # Risk factor summary
    st.subheader("⚕️ Key Risk Factors Present")
    risk_factors = []
    if age >= 60:           risk_factors.append(f"🔴 Age ≥ 60 ({age} years)")
    if hypertension == 1:   risk_factors.append("🔴 Hypertension")
    if heart_disease == 1:  risk_factors.append("🔴 Heart Disease")
    if avg_glucose > 140:   risk_factors.append(f"🔴 High Glucose ({avg_glucose:.1f} mg/dL)")
    if bmi > 30:            risk_factors.append(f"🟡 Obesity (BMI = {bmi:.1f})")
    if smoking == "smokes": risk_factors.append("🟡 Active Smoker")

    if risk_factors:
        for rf in risk_factors:
            st.markdown(f"- {rf}")
    else:
        st.markdown("✅ No major risk factors identified from the selected features.")

st.divider()
st.markdown("""
### ℹ️ About This Model

| Property | Value |
|----------|-------|
| Algorithm | Random Forest (Tuned — GridSearchCV) |
| Dataset | Kaggle Stroke Prediction — 5,110 patients |
| Feature Selection | SelectKBest — ANOVA F-test (Top 8 features) |
| Class Imbalance | Handled with `class_weight='balanced'` |
| Validation | Stratified 10-Fold Cross-Validation |
| New Feature | `glucose_bmi_ratio` (Feature Engineering) |
| Evaluation Metrics | Accuracy, Precision, Recall, F1, ROC-AUC |

*⚠️ This tool is for educational purposes only and does not replace medical diagnosis.*

*Final Project — Machine Learning Course*
""")
