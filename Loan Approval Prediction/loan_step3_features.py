import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load cleaned dataset
df = pd.read_csv("loan_data_cleaned.csv")

# Step 1: Encode categorical columns using LabelEncoder
cat_cols = ['gender', 'married', 'dependents', 'education', 'self_employed', 'property_area', 'loan_status']
le = LabelEncoder()

for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Step 2: Feature & target split
X = df.drop('loan_status', axis=1)  # Features
y = df['loan_status']               # Target

# Optional: save the processed version
df.to_csv("loan_data_processed.csv", index=False)
print("✅ Feature engineering done. Saved as 'loan_data_processed.csv'")
print("\nX shape:", X.shape)
print("y value counts:\n", y.value_counts())

