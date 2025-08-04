# Predictive-Maintenance

# 🔧 Predictive Maintenance Using Stacking Model

A machine learning project that predicts equipment failures before they occur using sensor and operational data. This project uses ensemble learning (stacking) to improve classification accuracy across multiple failure types.

---

## 📌 Problem Statement

Predictive maintenance enables proactive repairs of industrial machinery before failure, minimizing unplanned downtime. The goal of this project is to build a high-accuracy classification model that predicts potential machine failures like:

- Tool wear
- Power failure
- Overstrain
- Heat dissipation
- No failure

---

## 📁 Dataset

**Source:** [Kaggle - Machine Predictive Maintenance Classification](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification)

The dataset includes:

| Column Name | Description |
|-------------|-------------|
| `UDI` | Unique identifier |
| `Product ID` | ID of the machine |
| `Type` | Type of product (L/M/H) |
| `Air temperature [K]` | Air temperature in Kelvin |
| `Process temperature [K]` | Internal process temperature |
| `Rotational speed [rpm]` | Rotational speed in revolutions per minute |
| `Torque [Nm]` | Torque measurement |
| `Tool wear [min]` | Minutes of tool wear |
| `Target` | Failure type (multi-class) |

---

## 🛠️ Tech Stack

- **Python 3.10**
- **Pandas, NumPy, Matplotlib, Seaborn**
- **Scikit-Learn**
- **XGBoost**
- **Joblib** (for model saving)
- **IBM Cloud** (for deployment)

---

## 🧠 Machine Learning Workflow

1. **Data Preprocessing**
   - Label encoding for categorical features
   - Handling missing values
   - Feature scaling

2. **Exploratory Data Analysis (EDA)**
   - Correlation matrix
   - Distribution plots
   - Class imbalance analysis

3. **Model Building**
   - Base models: RandomForest, XGBoost, ExtraTrees
   - Meta-learner: Logistic Regression
   - Final: Stacking Classifier

4. **Evaluation Metrics**
   - Accuracy
   - Confusion Matrix
   - Classification Report (Precision, Recall, F1)

---

## 🎯 Results

| Model              | Accuracy |
|--------------------|----------|
| Random Forest      | 98.7%    |
| Extra Trees        | 98.8%    |
| XGBoost            | 98.6%    |
| **Stacking Model** | **99.55%** ✅ |

---

## 📦 Model Deployment on IBM Cloud (Optional)

To deploy the trained model using IBM Watson Machine Learning:

### 1. Save the model

```python
import joblib
joblib.dump(stacking_model, "stacking_model.pkl")
