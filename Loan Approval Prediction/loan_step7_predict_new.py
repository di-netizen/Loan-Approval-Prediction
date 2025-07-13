import pandas as pd
import joblib

# 🔄 Load the trained model
model = joblib.load('loan_approval_model.pkl')

# ✅ Simulate new user input (values must match processed structure)
# You can customize this input!
new_data = pd.DataFrame([{
    'gender': 1,              # Male
    'married': 1,             # Yes
    'dependents': 0,          # 0
    'education': 0,           # Graduate
    'self_employed': 0,       # No
    'applicantincome': 5000,
    'coapplicantincome': 2000,
    'loanamount': 150,
    'loan_amount_term': 360,
    'credit_history': 1.0,
    'property_area': 2        # Urban
}])

# ⚠️ Make sure columns are in same order as during training
df = pd.read_csv("loan_data_processed.csv")
X = df.drop("loan_status", axis=1)
new_data = new_data[X.columns]

# 🧠 Predict
prediction = model.predict(new_data)[0]
result = 'Approved ✅' if prediction == 1 else 'Not Approved ❌'

print("🔍 Prediction result:", result)
