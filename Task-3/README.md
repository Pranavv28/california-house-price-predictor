# 🏠 California House Price Predictor
### AI/ML Internship — Maincrafts Technology | Pranav Lakhe | SIT Nagpur

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Maincrafts](https://img.shields.io/badge/Internship-Maincrafts%20Technology-0d47a1)](https://maincrafts.com)

> A complete 3-task AI/ML internship project building an end-to-end house price prediction system — from a basic Linear Regression baseline all the way to a cross-validated, hyperparameter-tuned production-ready model.

---

## 📌 Project Overview

This repository documents the full ML lifecycle across 3 progressive tasks using the **California Housing Dataset** (1990 U.S. Census, 20,640 records):

| Task | Focus | Best Model | R² |
|------|-------|------------|-----|
| Task 1 | Linear Regression baseline | Linear Regression | 0.6634 |
| Task 2 | Feature Engineering + Model Comparison | Decision Tree (depth=5) | 0.8360 |
| **Task 3** | **Cross-Validation + Hyperparameter Tuning** | **Tuned Decision Tree** | **0.8456** |

**Total improvement: +18.2 percentage points in R² from Task 1 to Task 3.**

---

## 🗂️ Repository Structure

```
california-house-price-predictor/
│
├── 📁 Task-1/
│   ├── task1_ml_linear_regression.ipynb   # EDA → Preprocessing → Training → Evaluation
│   ├── task1_report.pdf                   # Project report
│   ├── linear_regression_model.pkl        # Saved model
│   └── predict_house_price.py             # CLI predictor script
│
├── 📁 Task-2/
│   ├── AI_ML_Task2_Model_Comparison.ipynb # Feature engineering + 3 model comparison
│   ├── task2_report.pdf                   # Standard report
│   ├── task2_report_ieee.pdf              # IEEE research paper format
│   └── best_model_task2.pkl              # Saved best model (Decision Tree)
│
├── 📁 Task-3/
│   ├── AI_ML_Task3_Model_Validation_Tuning.ipynb  # Overfitting + CV + GridSearchCV
│   ├── task3_report.pdf                   # Standard report
│   ├── task3_report_ieee.pdf              # IEEE research paper format
│   ├── best_model_task3.pkl              # Saved tuned model
│   └── plots/                            # All Task 3 visualizations
│
├── 🌐 app.py                              # Task 1 Streamlit web app
├── 🌐 app2.py                             # Task 2 Streamlit web app (model comparison)
├── 🌐 app3.py                             # Task 3 Streamlit web app (overfitting + tuning)
│
├── plots/                                 # Task 1 & 2 visualizations
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/Pranavv28/california-house-price-predictor.git
cd california-house-price-predictor
pip install -r requirements.txt
```

### 2. Run the Streamlit Apps
```bash
streamlit run app.py    # Task 1 — Basic predictor
streamlit run app2.py   # Task 2 — Model comparison
streamlit run app3.py   # Task 3 — Overfitting analysis + tuned predictor
```

### 3. Run the Notebooks
```bash
jupyter notebook
```
Open any notebook from Task-1, Task-2, or Task-3 folders.

---

## 📋 Task Breakdown

### ✅ Task 1 — Linear Regression Baseline

**Objective:** Introduce the full ML lifecycle on a real dataset.

**What was done:**
- Loaded California Housing Dataset via scikit-learn
- Performed EDA: correlation heatmap, feature distributions, missing value check
- Preprocessed with StandardScaler (no data leakage)
- Trained LinearRegression with 80/20 train-test split
- Evaluated with MAE, RMSE, R²
- Built Streamlit web app + CLI predictor

**Results:**

| Metric | Value |
|--------|-------|
| MAE | 0.4607 (~$46,000) |
| RMSE | 0.5571 |
| **R²** | **0.6634** |

**Key Finding:** MedInc (median income) is the strongest predictor with a standardized coefficient of 0.73.

---

### ✅ Task 2 — Feature Engineering & Model Comparison

**Objective:** Improve the baseline through feature engineering and algorithm comparison.

**What was done:**
- Engineered 4 new features from domain knowledge:
  - `RoomsPerPerson` = AveRooms / AveOccup
  - `BedroomRatio` = AveBedrms / AveRooms
  - `LogPopulation` = log(1 + Population)
  - `IncomePerRoom` = MedInc / AveRooms
- Compared 3 algorithms: Linear Regression, Ridge Regression, Decision Tree
- Used 5-fold cross-validation for reliable evaluation
- Built model comparison Streamlit app

**Results:**

| Model | MAE | RMSE | R² | CV R² |
|-------|-----|------|----|-------|
| Linear Regression | 0.4593 | 0.5555 | 0.6653 | 0.6617 |
| Ridge Regression | 0.4593 | 0.5555 | 0.6653 | 0.6617 |
| **Decision Tree (depth=5)** | **0.2716** | **0.3889** | **0.8360** | **0.8374** |

**Key Finding:** Decision Tree outperforms linear models by 17 percentage points in R² — confirming non-linear relationships in the data.

---

### ✅ Task 3 — Model Validation, Overfitting Control & Hyperparameter Tuning

**Objective:** Build a reliable, production-ready model using proper validation and tuning.

**What was done:**
- Detected overfitting: untuned Decision Tree had Train R²=1.0, Test R²=0.74 (gap=0.26)
- Applied 5-fold cross-validation to all models with std deviation analysis
- Used GridSearchCV over 36 parameter combinations:
  - `max_depth` ∈ {3, 5, 7, 10}
  - `min_samples_split` ∈ {2, 5, 10}
  - `min_samples_leaf` ∈ {1, 2, 4}
- Best params: `max_depth=7, min_samples_split=10, min_samples_leaf=2`
- Built overfitting curve + live prediction Streamlit app

**Results:**

| Model | MAE | RMSE | Test R² | Train R² | Gap |
|-------|-----|------|---------|----------|-----|
| Linear Regression (T1) | 0.4593 | 0.5555 | 0.6653 | 0.6673 | 0.002 |
| DT depth=5 (T2) | 0.2716 | 0.3889 | 0.8360 | 0.8716 | 0.035 |
| **Tuned DT (T3)** | **0.2632** | **0.3773** | **0.8456** | **0.8716** | **0.026** |

**Key Finding:** GridSearchCV reduced the overfitting gap from 0.259 (untuned) to 0.026 (tuned) while improving test R² to 0.8456.

---

## 📊 Key Visualizations

| Task 1 — Correlation Heatmap | Task 1 — Actual vs Predicted |
|---|---|
| ![heatmap](plots/heatmap.png) | ![avp](plots/actual_vs_pred.png) |

| Task 2 — Model Comparison | Task 2 — Feature Importances |
|---|---|
| ![mc](plots/model_comparison.png) | ![fi](plots/coefficients.png) |

| Task 3 — Overfitting Analysis | Task 3 — Train vs Test Gap |
|---|---|
| ![ov](Task-3/plots/overfitting_analysis.png) | ![gap](Task-3/plots/train_test_gap.png) |

---

## 🔬 Full ML Workflow

```
Raw Data (sklearn California Housing)
        │
        ▼
Task 1: Basic ML Pipeline
  EDA → StandardScaler → LinearRegression → MAE/RMSE/R²
        │
        ▼
Task 2: Feature Engineering + Model Comparison
  4 New Features → LinearReg vs Ridge vs DecisionTree → Cross-Validation
        │
        ▼
Task 3: Validation + Tuning
  Overfitting Detection → 5-Fold CV → GridSearchCV (36 combos) → Best Model
        │
        ▼
Final Tuned Model
  R²=0.8456 | MAE=~$26,000 | Gap=0.026 | Production-Ready
```

---

## 💡 Improvement Roadmap

- [x] Linear Regression baseline
- [x] Feature Engineering (4 new features)
- [x] Multi-model comparison
- [x] Cross-validation
- [x] GridSearchCV hyperparameter tuning
- [ ] Random Forest / XGBoost (expected R² > 0.87)
- [ ] RandomizedSearchCV for faster tuning
- [ ] SHAP values for model explainability
- [ ] Docker deployment

---

## 📦 Dependencies

```
pandas · numpy · scikit-learn · matplotlib · seaborn · streamlit · jupyter · joblib
```

Full list: [`requirements.txt`](requirements.txt)

---

## 👤 Author

**Pranav Lakhe**
B.Tech Student · Symbiosis Institute of Technology, Nagpur
AI/ML Intern @ Maincrafts Technology
🔗 [GitHub](https://github.com/Pranavv28)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">
  <i>Built as part of the Maincrafts Technology AI/ML Internship · Tasks 1, 2 & 3</i>
</p>
