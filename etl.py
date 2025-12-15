# etl.py
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Create necessary directories
os.makedirs('out', exist_ok=True)

# Load the dataset
DATA_PATH = 'data\\archive(1)\\diabetes_prediction_dataset.csv'  # Adjust if needed
df = pd.read_csv(DATA_PATH)

print(f"Original shape: {df.shape}")

### Schema Validation ###
expected_columns = [
    'gender', 'age', 'hypertension', 'heart_disease', 'smoking_history',
    'bmi', 'HbA1c_level', 'blood_glucose_level', 'diabetes'
]
if list(df.columns) != expected_columns:
    raise ValueError(f"Schema mismatch. Expected columns: {expected_columns}")

print("Schema validation passed.")

### Cleaning & Quarantine ###
# Remove duplicates
df = df.drop_duplicates()
print(f"After removing duplicates: {df.shape}")

# Initialize quarantine dataframe
quarantine = pd.DataFrame()

# Invalid gender values (only Male, Female, Other are valid in this dataset)
valid_genders = ['Male', 'Female', 'Other']
invalid_gender = df[~df['gender'].isin(valid_genders)]
if not invalid_gender.empty:
    quarantine = pd.concat([quarantine, invalid_gender])
    df = df[df['gender'].isin(valid_genders)]

# Invalid numeric values (age < 0 or bmi <= 0)
invalid_numeric = df[(df['age'] < 0) | (df['bmi'] <= 0)]
if not invalid_numeric.empty:
    quarantine = pd.concat([quarantine, invalid_numeric])
    df = df[(df['age'] >= 0) & (df['bmi'] > 0)]

# Drop any remaining NaNs (dataset is clean, but just in case)
df = df.dropna()

# Save quarantined rows
quarantine.drop_duplicates().to_csv('quarantine.csv', index=False)
print(f"Quarantined {len(quarantine)} rows → quarantine.csv")

### Encoding ###
df_encoded = pd.get_dummies(
    df,
    columns=['gender', 'smoking_history'],
    drop_first=True  # Avoid multicollinearity
)

### Z-score Scaling ###
numeric_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
scaler = StandardScaler()
df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])

### Stratified Train/Test Split ###
train_df, test_df = train_test_split(
    df_encoded,
    test_size=0.2,
    stratify=df_encoded['diabetes'],
    random_state=42
)

# Save clean data and splits
df_encoded.to_csv('clean_diabetes.csv', index=False)
train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)

print("ETL completed successfully!")
print(f"Clean dataset: {df_encoded.shape}")
print(f"Train: {train_df.shape}, Test: {test_df.shape}")