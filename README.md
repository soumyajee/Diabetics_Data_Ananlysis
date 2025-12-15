🩺 Test B — Diabetes Data Engineering & Analysis Challenge
📌 Overview

This project implements an end-to-end data validation, ETL, and exploratory data analysis pipeline for the Diabetes Prediction Dataset.

The solution strictly follows the assessment requirements and produces:

A clean, analysis-ready dataset

Quarantined invalid records

Statistical summaries

Correlation insights

Risk group analysis

Visualizations

Multicollinearity checks

All outputs are generated using pure Python scripts for reproducibility.
📂 Dataset

Source: Diabetes Prediction Dataset
Target Variable: diabetes

0 → No diabetes

1 → Diabetes
Original Columns
gender, age, hypertension, heart_disease, smoking_history,
bmi, HbA1c_level, blood_glucose_level, diabetes
📁 Project Structure
DIABETICS_TEST/
│
├── data_archive(1)/
│   └── diabetes_prediction_dataset.csv   # Raw dataset
│
├── etl.py                                 # Data validation & ETL pipeline
├── analysis.py                            # Exploratory data analysis
│
├── clean_diabetes.csv                     # Cleaned dataset (final)
├── quarantine.csv                         # Invalid / quarantined records
│
├── train.csv                              # Stratified train split
├── test.csv                               # Stratified test split
│
├── correlation.json                       # Feature ↔ diabetes correlations
├── risk_groups.csv                       # Risk cohort statistics
├── multicollinearity.txt                 # High correlation warnings
│
├── out/
│   └── plots/
│       ├── hist_age.png
│       ├── hist_bmi.png
│       ├── hist_HbA1c_level.png
│       ├── hist_blood_glucose_level.png
│       ├── box_age_by_diabetes.png
│       ├── box_bmi_by_diabetes.png
│       ├── box_HbA1c_level_by_diabetes.png
│       ├── box_blood_glucose_level_by_diabetes.png
│       ├── smoking_vs_diabetes.png
│       └── correlation_heatmap.png
│
└── Readme.md
⚙️ Part 1 — Data Validation & ETL

Implemented in etl.py

✔ Schema Validation

Ensures all expected columns are present

Validates correct data types

Detects invalid categorical values

✔ Cleaning & Quarantine

Rows with missing critical fields or invalid values are:

Removed from main dataset

Saved separately in quarantine.csv

Remaining rows form the clean dataset

✔ Encoding & Scaling

Categorical features encoded

Numeric features scaled using Z-score normalization

✔ Stratified Splits

Dataset split into:

train.csv

test.csv

Stratified by diabetes to preserve class distribution
📊 Part 2 — Exploratory Data Analysis (EDA)

Implemented in analysis.py

1️⃣ Summary Statistics

Computed for numeric features:

age

bmi

HbA1c_level

blood_glucose_level

Reported metrics:
| Feature | Mean | Std | Min | Median | Max | % Missing |

Also reported:

Male vs Female count

Smoking history distribution

Diabetes prevalence (%)
2️⃣ Correlation & Feature Insights

Pearson correlation computed between each numeric feature and diabetes

Results sorted by absolute correlation strength

Saved as:
{
  "HbA1c_level": 0.47,
  "blood_glucose_level": 0.42,
  "bmi": 0.29,
  "age": 0.21
}
Correlation heatmap saved in /out/plots/
3️⃣ Risk Group Statistics

Dataset split into meaningful cohorts:

Cohort	Condition
Elderly	age ≥ 60
Overweight	BMI ≥ 30
Hypertension	hypertension = 1
Heart Disease	heart_disease = 1
High Glucose	blood_glucose_level ≥ 180
Smokers	smoking_history ∈ {current, ever, former}

For each cohort:

Sample size (N)

Diabetes prevalence (%)

Saved as:
📄 risk_groups.csv
4️⃣ Feature Distributions

Saved to /out/plots/

✔ Histograms:

Age

BMI

HbA1c level

Blood glucose level

✔ Boxplots (grouped by diabetes):

All numeric features

✔ Bar chart:

Smoking history vs diabetes prevalence
5️⃣ Multicollinearity Check

Correlation matrix computed for numeric predictors

Absolute correlation > 0.8 flagged

Warnings printed and saved