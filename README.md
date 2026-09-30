# CAPSTONE-PROJECT-
# Predicting Student Depression with Machine Learning

*Data Science Capstone Project | Project Documentation (README)*

| | |
|---|---|
| **Author** | Mpolokeng Majake (202402151) |
| **Institution** | Sol Plaatje University |
| **GitHub repository** | https://github.com/YOUR-USERNAME/Depression-Prediction |

A machine learning project that uses lifestyle and academic factors to predict depression in university students, so that at-risk students can be identified and supported earlier.

## Problem Statement

Depression among university students often goes unnoticed until it harms academic performance and wellbeing. Universities have no data-driven way to identify at-risk students early, so support arrives too late. This project builds a model that predicts student depression from lifestyle and academic data.

## Objectives

1. Explore the data to find patterns between lifestyle, academic factors and depression.
2. Build a Random Forest classifier to predict whether a student is depressed.
3. Benchmark it against Decision Tree and Logistic Regression using Accuracy, F1 and ROC-AUC.
4. Identify the strongest predictors and save the model for reuse.

## Methodology

The project follows **CRISP-DM**: business understanding, data understanding, data preparation, modelling, evaluation and deployment.

**Pipeline:** Data Loading → Data Cleaning → EDA → Preprocessing → Model Training → Evaluation → Feature Importance → Model Deployment

## Dataset

| File | Description | Size |
|---|---|---|
| `student_depression_dataset.csv` | Student records with lifestyle and academic factors | 27,901 rows |
| `lifestyle_records.csv` | Complementary lifestyle dataset | 2,000 rows |

**Sources:**
- Lifestyle dataset: https://www.kaggle.com/datasets/steve1215rogg/student-lifestyle-dataset
- Student depression dataset: https://www.kaggle.com/code/saifeldeenmohammedm/student-depression-dataset

**Target variable:** Depression (1 = depressed: 58.5%, 0 = not depressed: 41.5%)

### Features (17)

- **Numerical:** Age, Academic Pressure, Work Pressure, CGPA, Study Satisfaction, Job Satisfaction, Work/Study Hours, Financial Stress
- **Categorical:** Gender, City, Profession, Degree, Sleep Duration, Dietary Habits, Suicidal Thoughts History, Family History of Mental Illness

### Data cleaning

- Dropped the non-predictive ID column.
- Converted 3 '?' values in Financial Stress to NaN and imputed with the median (3.0).
- Stripped stray quote characters from Profession, Sleep Duration and Degree.
- Result: 27,901 rows, 17 features, 0 missing values.

### Preprocessing

Categorical columns were label-encoded; numerical features were standard-scaled for Logistic Regression only. The data was split 80/20 with stratification (22,320 train / 5,581 test).

## Models and Results

Three models were trained with class_weight='balanced' to handle the 58.5% / 41.5% class imbalance and compared using 5-fold stratified cross-validation.

| Model | 5-fold CV F1 |
|---|---|
| Random Forest (200 trees) | 0.8657 |
| Logistic Regression | 0.86 |
| Decision Tree (max_depth=10) | 0.84 |

**Performance on the test set (5,581 unseen students):**

| Model | Accuracy | F1-Score | ROC-AUC |
|---|---|---|---|
| Random Forest (primary model) | 0.8377 | 0.8605 | 0.9156 |
| Decision Tree | 0.8147 | 0.8387 | 0.8630 |
| Logistic Regression | 0.8404 | 0.8610 | 0.9173 |

Random Forest and Logistic Regression perform almost identically, and both clearly outperform the Decision Tree. Random Forest was chosen as the primary model because it handles mixed data types and provides feature importance rankings.

### Top predictors (feature importance)

1. Suicidal thoughts history: 26.4%
2. Academic pressure: 19.1%
3. Financial stress: 10.5%

### Figures

**Confusion Matrix**

![Confusion Matrix](figures/confusion_matrix.png)

**ROC Curve**

![ROC Curve](figures/roc_curve.png)

**Feature Importance**

![Feature Importance](figures/feature_importance.png)

## Actionable Insights

- **Early-warning screening:** flag students with a history of suicidal thoughts for counsellor follow-up.
- **Manage academic pressure:** review workload and assessment clustering.
- **Financial support:** promote bursaries and emergency aid.
- **Promote sleep health:** about 72% of students sleeping under 5 hours are depressed, so run sleep-awareness programmes.

## Limitations

- The data is largely self-reported, so responses may be subjective or inaccurate.
- "Suicidal thoughts history" is a very strong predictor and sits close to the outcome being predicted. It should be interpreted carefully, and the model's performance without it is worth checking.
- The model shows association, not causation.
- Results come from one dataset and may not generalise to all universities or populations.
- This model is a screening aid to support counsellors. It is not a diagnostic tool and does not replace professional assessment.

## Repository Structure

```
Depression-Prediction/
├── README.md
├── notebooks/
│   └── Depression_Prediction.ipynb
├── data/
│   ├── student_depression_dataset.csv
│   └── lifestyle_records.csv
├── models/
│   └── random_forest_depression_model.pkl
├── figures/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── feature_importance.png
├── requirements.txt
└── .gitignore
```

## Tools

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, joblib, Jupyter
