# 🤖 Task 2 — Feature Engineering, Model Optimization & Performance Comparison

**Maincrafts Technology · AI/ML Internship · Pranav Lakhe | SIT Nagpur**

---

## 📌 Objective

Improve the Task 1 Linear Regression baseline by applying **feature engineering** and comparing **3 regression algorithms** to find the best-performing model.

---

## 📁 Files

| File | Description |
|------|-------------|
| `AI_ML_Task2_Model_Comparison.ipynb` | Main notebook — feature engineering, training, comparison |
| `task2_report.pdf` | Standard project report |
| `task2_report_ieee.pdf` | IEEE research paper format |
| `best_model_task2.pkl` | Saved best model (Decision Tree, joblib) |

---

## 🔧 Feature Engineering

4 new features derived from domain knowledge:

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `RoomsPerPerson` | AveRooms / AveOccup | Living space per resident |
| `BedroomRatio` | AveBedrms / AveRooms | Bedroom-heavy = lower value |
| `LogPopulation` | log(1 + Population) | Reduces right-skew |
| `IncomePerRoom` | MedInc / AveRooms | Economic density per space |

---

## 📊 Results

| Model | MAE | RMSE | R² | CV R² |
|-------|-----|------|----|-------|
| Linear Regression | 0.4593 | 0.5555 | 0.6653 | 0.6617 |
| Ridge Regression | 0.4593 | 0.5555 | 0.6653 | 0.6617 |
| ✅ **Decision Tree (depth=5)** | **0.2716** | **0.3889** | **0.8360** | **0.8374** |

**Best Model: Decision Tree — R² improved by +17 points over Task 1 baseline.**

---

## 🧠 Why Decision Tree Wins

- Captures **non-linear relationships** between income, location, and price
- `max_depth=5` prevents overfitting while retaining complexity
- CV R²=0.8374 confirms strong generalization
- MAE of ~$27,000 vs ~$46,000 for linear models

---

## ▶️ Run

```bash
jupyter notebook AI_ML_Task2_Model_Comparison.ipynb
```

---

*Part of the Maincrafts Technology AI/ML Internship · [Main Repo](../README.md)*
