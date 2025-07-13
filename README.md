# 🏦 Loan Approval Prediction

This project walks through a full machine learning pipeline — from generating synthetic loan data to building a model that can predict whether a loan will be approved or not. It’s designed to simulate a real-world workflow, including data creation, cleaning, visualization, modeling, evaluation, and deployment.

---

## 🚀 What This Project Does

- Creates a synthetic loan dataset
- Cleans and preprocesses the data
- Explores data visually using Seaborn and Matplotlib
- Encodes categorical values with LabelEncoder
- Trains and evaluates two ML models:
  - Logistic Regression
  - Random Forest Classifier
- Shows model accuracy, confusion matrix, and classification report
- Displays feature importance for interpretability
- Saves the trained model using `joblib`
- Uses the saved model to predict loan approval on new data

---

## 🗂️ Project Structure

```bash
📁 loan-approval-prediction/
├── loan_data.csv                # Raw synthetic data
├── loan_data_cleaned.csv        # Cleaned dataset
├── loan_data_processed.csv      # Label-encoded dataset
├── loan_approval_model.pkl      # Saved Random Forest model
├── generate_data.py             # Generates and cleans data
├── eda.py                       # Exploratory Data Analysis
├── model_training.py            # Model training and evaluation
├── predict_new_data.py          # Predict loan status on new input
