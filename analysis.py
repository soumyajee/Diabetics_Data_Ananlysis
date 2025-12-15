# analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# Create output directory for plots
os.makedirs('out/plots', exist_ok=True)

# IMPORTANT: Load the ORIGINAL raw data for EDA (not the cleaned/encoded version)
df = pd.read_csv('clean_diabetes.csv')  # Fixed path to raw data

numeric_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']

### 1. Summary Statistics ###
summary = df[numeric_cols].agg(['mean', 'std', 'min', 'median', 'max']).round(2)
summary.loc['% Missing'] = (df[numeric_cols].isnull().mean() * 100).round(2)
summary = summary.T
summary.reset_index(inplace=True)
summary.rename(columns={'index': 'Feature'}, inplace=True)

print("=== Summary Statistics ===")
print(summary.to_string(index=False))

# Additional reports
print("\n=== Gender Distribution ===")
print(df['gender_Male'].value_counts())
print(df['gender_Other'].value_counts())

print("\n=== Smoking History Counts ===")
print(df['smoking_history_current'].value_counts())


diabetes_prev = df['diabetes'].mean() * 100
print(f"\nDiabetes Prevalence: {diabetes_prev:.2f}%")

### 2. Correlation & Feature Insights ###
corr_with_diabetes = df[numeric_cols + ['diabetes']].corr()['diabetes'] \
                     .drop('diabetes') \
                     .sort_values(ascending=False)

print("\n=== Correlations with Diabetes (sorted) ===")
print(corr_with_diabetes.round(4))

# Save to JSON
with open('correlation.json', 'w') as f:
    json.dump(corr_with_diabetes.to_dict(), f, indent=4)

# Optional: Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df[numeric_cols + ['diabetes']].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('out/plots/correlation_heatmap.png')
plt.close()

### 3. Risk Group Statistics ###
risk_groups = []

cohorts = [
    ('Elderly', 'age >= 60', df['age'] >= 60),
    ('Overweight', 'BMI >= 30', df['bmi'] >= 30),
    ('Hypertension', 'hypertension == 1', df['hypertension'] == 1),
    ('Heart Disease', 'heart_disease == 1', df['heart_disease'] == 1),
    ('High Glucose', 'blood_glucose_level >= 180', df['blood_glucose_level'] >= 180),
    ('Smokers', 'smoking_history in {current, ever, former}',
     df['smoking_history_current'].isin(['current', 'ever', 'former']))
]

for name, condition_str, mask in cohorts:
    subset = df[mask]
    n = len(subset)
    diabetes_pct = subset['diabetes'].mean() * 100 if n > 0 else 0
    risk_groups.append({
        'Cohort': name,
        'Condition': condition_str,
        'N': n,
        'Diabetes %': round(diabetes_pct, 2)
    })

risk_df = pd.DataFrame(risk_groups)
risk_df.to_csv('risk_groups.csv', index=False)

print("\n=== Risk Group Statistics ===")
print(risk_df.to_string(index=False))

### 4. Feature Distributions ###
# Histograms
for col in numeric_cols:
    plt.figure(figsize=(8, 5))
    df[col].hist(bins=50, edgecolor='black')
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f'out/plots/hist_{col}.png')
    plt.close()

# Boxplots by diabetes
for col in numeric_cols:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='diabetes', y=col, data=df)
    plt.title(f'{col} by Diabetes Status')
    plt.tight_layout()
    plt.savefig(f'out/plots/box_{col}_by_diabetes.png')
    plt.close()

# Smoking history vs diabetes prevalence
smoking_prev = df.groupby('smoking_history_current')['diabetes'].mean().sort_values() * 100
plt.figure(figsize=(8, 5))
smoking_prev.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Diabetes Prevalence by Smoking History')
plt.ylabel('Diabetes %')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('out/plots/smoking_vs_diabetes.png')
plt.close()

### 5. Multicollinearity Check ###
corr_matrix = df[numeric_cols].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr = [(i, j, corr_matrix.loc[i, j])
             for i, j in zip(*np.where(upper > 0.8))]

with open('multicollinearity.txt', 'w') as f:
    if high_corr:
        for col1, col2, val in high_corr:
            warning = f"⚠️ High correlation detected between {col1} and {col2} (r={val:.2f})\n"
            print(warning.strip())
            f.write(warning)
    else:
        msg = "No high correlations (>0.8) detected.\n"
        print(msg.strip())
        f.write(msg)

print("\nAll analysis outputs saved!")