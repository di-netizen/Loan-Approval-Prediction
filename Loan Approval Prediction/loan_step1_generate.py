import pandas as pd
import numpy as np

# Set seed
np.random.seed(42)

# Number of records
n = 1000

# Generate synthetic features
df = pd.DataFrame({
    'Gender': np.random.choice(['Male', 'Female'], n),
    'Married': np.random.choice(['Yes', 'No'], n),
    'Dependents': np.random.choice(['0', '1', '2', '3+'], n),
    'Education': np.random.choice(['Graduate', 'Not Graduate'], n),
    'Self_Employed': np.random.choice(['Yes', 'No'], n),
    'ApplicantIncome': np.random.randint(1500, 25000, n),
    'CoapplicantIncome': np.random.randint(0, 10000, n),
    'LoanAmount': np.random.randint(50, 700, n),
    'Loan_Amount_Term': np.random.choice([360, 120, 240, 180], n),
    'Credit_History': np.random.choice([1.0, 0.0], n, p=[0.8, 0.2]),
    'Property_Area': np.random.choice(['Urban', 'Semiurban', 'Rural'], n),
    'Loan_Status': np.random.choice(['Y', 'N'], n, p=[0.75, 0.25])
})

# Save dataset
df.to_csv("loan_data.csv", index=False)
print("✅ Synthetic loan dataset saved as 'loan_data.csv'")

# Load and clean
df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
df.to_csv("loan_data_cleaned.csv", index=False)
print("✅ Cleaned dataset saved as 'loan_data_cleaned.csv'")


