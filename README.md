# 🧠 Stroke Prediction Using Machine Learning
- Name :Hala Amr Awad — ID: 221001189
## Project Description

This project predicts the risk of stroke in patients using clinical and demographic features. The goal is to answer: **Can we accurately predict stroke risk, and which factors matter most?**

The dataset contains **50,000 patient records** with **12 features** including age, BMI, glucose level, hypertension, heart disease, and more. A new feature (`glucose_bmi_ratio`) was engineered to capture compound cardiovascular risk. Three machine learning models were trained and compared, with a **Tuned Random Forest** achieving the best performance across all metrics.

The final model is deployed as a live web app using **Streamlit**.

---

## 🌐 Live Demo

🔗 Streamlit App: https://hala-awad-bfcz3jhwvppqpqurd8yc4z.streamlit.app/#stroke-risk-prediction-app 

---

## 📁 Dataset

- **Original Source:** The dataset link: https://github.com/halamr2004-byte/Stroke-Prediction-App-ML-project/blob/4f387c382ef178bdbb6476f5cfa5 
- **Size:** ~50,000 rows, 12 columns
- **Target variable:** `stroke` (0 = No Stroke, 1 = Stroke)
- **Stroke Positive Rate:** 4.87% (severely imbalanced dataset)

### Features Used
| Feature | Description |
|---|---|
| Age | Patient age |
| Gender | Male / Female |
| BMI | Body Mass Index |
| avg_glucose_level | Average blood glucose level |
| hypertension | 0 or 1 |
| heart_disease | 0 or 1 |
| work_type | Type of employment |
| Residence_type | Urban / Rural |
| smoking_status | Smoking history |
| ever_married | Yes / No |
| glucose_bmi_ratio ⭐ | Engineered feature: glucose ÷ BMI |

---

## 🎥 Demo Video

🔗 PowerPoint Link: https://github.com/halamr2004-byte/Stroke-Prediction-App-ML-project/blob/main/Stroke_Prediction_Presentation.pptx 

 Video link : https://github.com/halamr2004-byte/Stroke-Prediction-App-ML-project/blob/main/Meeting%20with%20Hala%20Amr%20Awad-20260603_134925-Meeting%20Recording.mp4

## ⚙️ How to Run the Project

### 1.  the repository Link 

https://github.com/halamr2004-byte/Stroke-Prediction-App-ML-project 


### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app locally
```bash
streamlit run app.py
```

---

## 📦 Required Libraries

```
pandas
numpy
scikit-learn
streamlit
matplotlib
seaborn
joblib
```

> Install all at once: `pip install -r requirements.txt`

---

## 🤖 Models Compared

| Model | Precision | Recall | AUC |

| Logistic Regression | ✅ ≥ 0.3 | ✅ ≥ 0.3 | Good |
| Random Forest | ✅ ≥ 0.3 | ✅ ≥ 0.3 | Better |
| SVM (RBF Kernel) | ✅ ≥ 0.3 | ✅ ≥ 0.3 | Good |
| **Tuned Random Forest ⭐** | **BEST** | **BEST** | **BEST** |

All models use `class_weight='balanced'` to handle class imbalance.

---

## 📊 Key Findings

- **Age** is the single most powerful predictor — stroke risk rises sharply after 60
- **glucose_bmi_ratio** (engineered feature) ranked in the **top 3 most important features**
- **Tuned Random Forest** (via GridSearchCV) outperformed all other models
- **10-Fold Stratified Cross-Validation** was used for reliable performance estimation

---

## 🚀 Deployment

The model is saved as `saved_model.pkl` and deployed on **Streamlit Cloud**.

- Model file: `saved_model.pkl`
- App entry point: `app.py`
- Dependencies: `requirements.txt`


## 📝 Notes

- The dataset is highly imbalanced (~95% no stroke, ~5% stroke), which was handled using `class_weight='balanced'`
- Feature engineering (`glucose_bmi_ratio`) was validated using SelectKBest and embedded (Random Forest) feature importance methods
- The `id` column was dropped as it has no predictive value
- Missing BMI values were filled with the median to avoid outlier sensitivity

Streamlit App: https://hala-awad-bfcz3jhwvppqpqurd8yc4z.streamlit.app/#stroke-risk-prediction-app
The dataset link: https://github.com/halamr2004-byte/Stroke-Prediction-App-ML-project/blob/4f387c382ef178bdbb6476f5cfa56094018dcb2e/synthetic_stroke_data.csv 
