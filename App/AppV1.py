import streamlit as st
import numpy as np
import pickle
import os

# loading files
def load_model(filename):
    base_dir = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(base_dir, filename)  # no "data" here
    with open(path, "rb") as f:
        return pickle.load(f)



scaler = load_model("Training Data/scaler.pkl")
rf_model = load_model("Training Data/rfmodel.pkl")
xgb_model = load_model("Training Data/xgmodel.pkl")
log_model = load_model("Training Data/lgmodel.pkl")


# load models
models = {
    'Logistic Regression': log_model,
    'Random Forest': rf_model,
    'XGBoost': xgb_model
    }
# Set threshold
THRESHOLD = 0.8

# Title
st.title("LendingClub Loan Defaulters Prediction App")

# Model Selection
model_choice = st.selectbox("Choose your preferred model", list(models.keys()))
model = models[model_choice]

# Input fields
st.header("Enter Applicant Details")
loan_amount = st.number_input("Loan Amount", min_value=0)
installment = st.number_input("Installment", min_value=0)
int_rate = st.number_input("Interest Rate", min_value=0)
ann_inc = st.number_input("Annual Income", min_value=0)
dti = st.number_input("Debt to Income Ratio", min_value= 0)


# Collect inputs
input_data = np.array([[installment, ann_inc, loan_amount, dti, int_rate]])
input_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict"):
    prob = model.predict_proba(input_scaled)[0][1]
    prediction = "Likely to Default" if prob > THRESHOLD else "Likely to Repay"

    st.subheader("Prediction Result")
    st.write(f"Model Used: {model_choice}")
    st.write(f"Probability of Default: {prob:.2f}")
    st.write(f"Final Decision: **{prediction}**")


