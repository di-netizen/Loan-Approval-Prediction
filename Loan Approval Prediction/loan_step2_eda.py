import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load cleaned data
df = pd.read_csv("loan_data_cleaned.csv")

# Countplot: Loan Status
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='loan_status')
plt.title("Loan Approval Distribution")
plt.tight_layout()
plt.show()

# Boxplot: Applicant Income vs Loan Status
plt.figure(figsize=(8, 4))
sns.boxplot(x='loan_status', y='applicantincome', data=df)
plt.title("Applicant Income by Loan Status")
plt.tight_layout()
plt.show()

# Boxplot: Loan Amount vs Loan Status
plt.figure(figsize=(8, 4))
sns.boxplot(x='loan_status', y='loanamount', data=df)
plt.title("Loan Amount by Loan Status")
plt.tight_layout()
plt.show()

# Countplot: Credit History
plt.figure(figsize=(6, 4))
sns.countplot(x='credit_history', hue='loan_status', data=df)
plt.title("Loan Status by Credit History")
plt.tight_layout()
plt.show()

# Heatmap: Correlation
numeric = df.select_dtypes(include=['float64', 'int64'])
plt.figure(figsize=(8, 6))
sns.heatmap(numeric.corr(), annot=True, cmap='Blues')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()
