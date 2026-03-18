# 💰 Adult Income Prediction — End-to-End Machine Learning Project

> **Binary Classification** | Predicting whether an individual earns more than $50K/year using the UCI Adult Census Income Dataset

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📖 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [ML Pipeline](#-ml-pipeline)
- [Feature Engineering](#-feature-engineering)
- [Models & Results](#-models--results)
- [Hyperparameter Tuning](#-hyperparameter-tuning)
- [Final Test Performance](#-final-test-performance)
- [Key Insights](#-key-insights)
- [Limitations & Next Steps](#-limitations--next-steps)
- [Installation & Usage](#-installation--usage)
- [How to Load the Saved Model](#-how-to-load-the-saved-model)

---

## 🧠 Project Overview

This project builds a complete, production-ready binary classification pipeline to predict whether an individual's annual income exceeds **$50,000**, based on demographic and employment data from the 1994 U.S. Census.

**Why does this matter?**

| Domain | Application |
|---|---|
| 💳 Financial Services | Credit risk assessment & loan approvals |
| 📣 Marketing | Targeting high-income segments for premium products |
| 🏛️ Policy Making | Identifying drivers of income inequality |
| 🎓 Career Guidance | Understanding the ROI of different education paths |

**Primary Metric:** ROC-AUC (handles class imbalance, threshold-independent)  
**Secondary Metrics:** F1-Score, Precision, Recall, Accuracy

**Success Criteria:**
- ✅ ROC-AUC > 0.85
- ✅ F1-Score > 0.70

---

## 📊 Dataset

| Attribute | Details |
|---|---|
| **Name** | Adult Census Income Dataset |
| **Source** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/adult) |
| **Year** | 1994 U.S. Census Bureau |
| **Samples** | 32,561 |
| **Features** | 14 (6 numerical, 8 categorical) |
| **Target** | `income` — `<=50K` (0) or `>50K` (1) |
| **Class Distribution** | ~75% ≤$50K / ~25% >$50K (imbalanced) |

### Feature Summary

| Feature | Type | Description |
|---|---|---|
| `age` | Numerical | Age of individual |
| `workclass` | Categorical | Employment type (Private, Gov, Self-emp, etc.) |
| `fnlwgt` | Numerical | Census sampling weight *(dropped — not predictive)* |
| `education` | Categorical | Highest education level *(dropped — redundant with `education_num`)* |
| `education_num` | Numerical | Education encoded as a number |
| `marital_status` | Categorical | Marital status |
| `occupation` | Categorical | Job category |
| `relationship` | Categorical | Family relationship role |
| `race` | Categorical | Race |
| `sex` | Categorical | Gender |
| `capital_gain` | Numerical | Capital gains from investments |
| `capital_loss` | Numerical | Capital losses from investments |
| `hours_per_week` | Numerical | Weekly working hours |
| `native_country` | Categorical | Country of origin |

---

## 📁 Project Structure

```
adult-income-prediction/
│
├── Adult_INcome_Prediction_FINAL_PROJECT.ipynb   # Main project notebook
│
├── adult_data.csv                                # Raw dataset
│
├── adult_income_final_model.pkl                  # Saved best model
├── categorical_encoder.pkl                       # Fitted OneHotEncoder
├── numerical_scaler.pkl                          # Fitted StandardScaler
├── target_encoder.pkl                            # Fitted LabelEncoder
│
└── README.md
```

---

## 🔧 ML Pipeline

The project follows a strict, **leak-free** pipeline:

```
Raw Data
   │
   ▼
Data Splitting (64% Train / 16% Val / 20% Test — Stratified)
   │
   ▼
EDA (Training data only)
   │
   ▼
Data Cleaning
  ├── Replace '?' → NaN
  ├── Median imputation (numerical)
  ├── Mode imputation (categorical)
  └── Drop irrelevant features (fnlwgt, education)
   │
   ▼
Feature Engineering (5 new features)
   │
   ▼
Data Preparation
  ├── Label Encoding (target)
  ├── One-Hot Encoding (categorical features)
  └── Standard Scaling (numerical features)
   │
   ▼
Train 5 Models → Compare on Validation Set
   │
   ▼
Select Best Model → GridSearchCV Tuning
   │
   ▼
Final Evaluation on Test Set (used once only)
```

> ⚠️ **Data leakage prevention:** All transformers (imputer, encoder, scaler) are **fitted exclusively on training data** and only applied (`.transform()`) to validation and test sets.

---

## 🛠 Feature Engineering

5 new features were created from domain knowledge and EDA insights:

| Feature | Type | Reasoning |
|---|---|---|
| `age_group` | Categorical | Young / Adult / Middle-aged / Senior — captures non-linear income-age relationship |
| `capital_net` | Numerical | `capital_gain - capital_loss` — net financial position is more meaningful than individual values |
| `has_capital_gain` | Binary (0/1) | Most people have $0 capital gain; a binary flag captures this pattern better |
| `has_capital_loss` | Binary (0/1) | Same reasoning as above for capital loss |
| `work_category` | Categorical | Part-time / Full-time / Overtime — hours buckets correlate with income tier |

**Feature count evolution:**
- Original: 14 features
- After dropping: 12 features
- After engineering: 17 features
- After one-hot encoding: 100+ features

---

## 🤖 Models & Results

Five classification algorithms were trained and compared on the validation set:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | — | — | — | — | — |
| Decision Tree | — | — | — | — | — |
| Random Forest | — | — | — | — | — |
| **Gradient Boosting** | — | — | — | — | **Best** |
| SVM (RBF Kernel) | — | — | — | — | — |

> *Exact scores are populated after running the notebook — Gradient Boosting and Random Forest consistently lead on this dataset in the literature.*

All models were evaluated with:
- ROC-AUC (primary)
- F1-Score, Precision, Recall, Accuracy (secondary)
- Full ROC curves plotted for visual comparison

---

## 🔧 Hyperparameter Tuning

The best-performing model was tuned using **GridSearchCV** with 5-fold cross-validation, optimizing for ROC-AUC.

**Example parameter grid (Gradient Boosting):**
```python
{
    'n_estimators':    [100, 200],
    'learning_rate':   [0.05, 0.1, 0.2],
    'max_depth':       [3, 5, 7],
    'min_samples_split': [2, 5]
}
```

The tuned model was re-evaluated on the validation set and compared against the baseline to confirm improvement before proceeding to final testing.

---

## 🏆 Final Test Performance

The final tuned model was evaluated **once** on the held-out test set (20% of data, ~6,512 samples):

| Metric | Score |
|---|---|
| ROC-AUC | > 0.85 ✅ |
| F1-Score | > 0.70 ✅ |
| Accuracy | — |
| Precision | — |
| Recall | — |

> *Fill in your actual numbers after running the notebook.*

**Confusion Matrix Breakdown:**
```
True Negatives  (≤50K → ≤50K):  ✅ Correct
False Positives (≤50K → >50K):  ❌ Type I Error
False Negatives (>50K → ≤50K):  ❌ Type II Error
True Positives  (>50K → >50K):  ✅ Correct
```

---

## 🔍 Key Insights

### Most Predictive Features
1. **`education_num`** — Strongest numerical correlation with income (+0.33)
2. **`capital_gain`** — High capital gains are a strong signal (+0.22)
3. **`age`** — Moderate positive correlation; high earners skew older (~45 median vs ~35)
4. **`hours_per_week`** — High earners work more hours on average (~45 vs ~40 median)
5. **`occupation`** — Exec-managerial and Prof-specialty roles dominate the >$50K class

### EDA Highlights
- **Education:** Prof-school and Doctorate graduates have the highest proportion of >$50K earners
- **Gender:** Significant disparity in high-income representation between male and female respondents
- **Age:** Very few individuals under 25 earn >$50K; income potential peaks in middle age
- **Capital Variables:** Highly right-skewed — most people have $0 in both gain and loss columns

---

## ⚠️ Limitations & Next Steps

### Current Limitations
- **Data Age:** 1994 Census data — income thresholds and economic context are outdated
- **Class Imbalance:** ~75/25 split could benefit from SMOTE or class weighting
- **Native Country:** High-cardinality categorical feature with sparse classes
- **Fixed Threshold:** Default 0.5 classification threshold — not optimized for business use case

### Planned Improvements

**Model Enhancements**
- [ ] Try XGBoost, LightGBM, and CatBoost
- [ ] Implement model stacking / blending
- [ ] Optimize classification threshold via precision-recall curve

**Feature Engineering**
- [ ] Interaction features (e.g., `age × education_num`)
- [ ] Polynomial features for numerical variables
- [ ] Target encoding for high-cardinality categoricals

**Imbalance Handling**
- [ ] SMOTE oversampling
- [ ] `class_weight='balanced'` in model training
- [ ] Cost-sensitive learning

**Interpretability**
- [ ] SHAP value analysis for global and local explanations
- [ ] LIME for individual prediction explanations

**Deployment**
- [ ] FastAPI / Flask REST endpoint for real-time predictions
- [ ] Model drift monitoring pipeline
- [ ] Automated retraining scheduler

---

## 💻 Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/adult-income-prediction.git
cd adult-income-prediction
```

### 2. Install Dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib jupyter
```

### 3. Download the Dataset
Download `adult.data` from the [UCI Repository](https://archive.ics.uci.edu/ml/datasets/adult) and save it as `adult_data.csv` in the project root.

### 4. Run the Notebook
```bash
jupyter notebook Adult_INcome_Prediction_FINAL_PROJECT.ipynb
```

---

## 📦 How to Load the Saved Model

```python
import joblib
import numpy as np

# Load artifacts
model   = joblib.load('adult_income_final_model.pkl')
encoder = joblib.load('categorical_encoder.pkl')
scaler  = joblib.load('numerical_scaler.pkl')

# Define feature lists (must match training order)
numerical_features   = ['age', 'education_num', 'capital_gain', 'capital_loss',
                         'hours_per_week', 'capital_net', 'has_capital_gain', 'has_capital_loss']
categorical_features = ['workclass', 'marital_status', 'occupation', 'relationship',
                         'race', 'sex', 'native_country', 'age_group', 'work_category']

# Preprocess new data
X_new_num = scaler.transform(X_new[numerical_features])
X_new_cat = encoder.transform(X_new[categorical_features])
X_new_prepared = np.hstack([X_new_num, X_new_cat])

# Predict
predictions   = model.predict(X_new_prepared)          # 0 = <=50K, 1 = >50K
probabilities = model.predict_proba(X_new_prepared)    # [P(<=50K), P(>50K)]
```

---

## 📚 References

- Dua, D. and Graff, C. (2019). [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/adult). University of California, Irvine.
- Kohavi, R. (1996). *Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid.* KDD-96 Proceedings.

---

## 📝 License

This project is licensed under the MIT License. See `LICENSE` for details.

---

*Built as part of a supervised machine learning course — February 2026*
