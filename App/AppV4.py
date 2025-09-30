import streamlit as st
import numpy as np
import pickle
import os

# ---------------------------
# Helper to load models safely
# ---------------------------
def load_model(filename):
    base_dir = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(base_dir, filename)
    with open(path, "rb") as f:
        return pickle.load(f)

# ---------------------------
# Load scaler and models
# ---------------------------
scaler = load_model("Training Data/scaler.pkl")
rf_model = load_model("Training Data/rfmodel.pkl")
xgb_model = load_model("Training Data/xgmodel.pkl")
log_model = load_model("Training Data/lgmodel.pkl")

models = {
    "Logistic Regression": log_model,
    "Random Forest": rf_model,
    "XGBoost": xgb_model,
}

THRESHOLD = 0.8

# ---------------------------
# Page Layout
# ---------------------------
st.set_page_config(page_title="Loan Default Prediction", layout="wide")
st.title(" LendingClub Loan Defaulters Prediction App")

# ---------------------------
# Sidebar - model selection + tracker
# ---------------------------
st.sidebar.header(" Model Selection")
model_choice = st.sidebar.selectbox("Choose your preferred model", list(models.keys()))
model = models[model_choice]

st.sidebar.markdown("---")
st.sidebar.header(" Process Tracker")

# ---------------------------
# Main Input Section
# ---------------------------
st.header("📋 Enter Applicant Details")

# 2x2 grid layout for inputs
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

with col1:
    loan_amount = st.number_input("Loan Amount", min_value=0, step=1000, format="%d")
with col2:
    installment = st.number_input("Installment", min_value=0, step=100, format="%d")
with col3:
    int_rate = st.number_input("Interest Rate (%)", min_value=0.0, step=0.1, format="%.2f")
with col4:
    ann_inc = st.number_input("Annual Income", min_value=0, step=1000, format="%d")

# Last input across full width
dti = st.number_input("Debt to Income Ratio", min_value=0.0, step=0.1, format="%.2f")

# ---------------------------
# Ensure strict input order
# ---------------------------
input_data = np.array([[loan_amount, installment, int_rate, ann_inc, dti]])

# Sidebar shows raw inputs
st.sidebar.subheader(" Raw Inputs")
st.sidebar.write({
    "loan_amount": loan_amount,
    "installment": installment,
    "int_rate": int_rate,
    "ann_inc": ann_inc,
    "dti": dti
})

# ---------------------------
# Run prediction instantly
# ---------------------------
if np.any(input_data > 0):
    input_scaled = scaler.transform(input_data)

    # Show scaled values in sidebar (for transparency)
    st.sidebar.subheader(" Scaled Inputs")
    st.sidebar.write(input_scaled.tolist())

    # Prediction
    prob = model.predict_proba(input_scaled)[0][1]
    prediction = " Likely to Default" if prob > THRESHOLD else "Likely to Repay"

    # Show prediction in main panel
    st.subheader(" Prediction Result")
    st.write(f"**Model Used:** {model_choice}")
    st.write(f"**Probability of Default:** {prob:.2%}")
    st.success(f"**Final Decision:** {prediction}")

    # Mirror prediction in sidebar
    st.sidebar.subheader(" Live Prediction")
    st.sidebar.write(f"**Probability:** {prob:.2%}")
    st.sidebar.write(f"**Decision:** {prediction}")
else:
    st.sidebar.info("Waiting for inputs... ")
