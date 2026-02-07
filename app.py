import streamlit as st
import pandas as pd
import pickle
import joblib
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Netflix Churn Predictor",
    page_icon="🎬",
    layout="centered"
)

# ---------------- NETFLIX CSS ----------------
st.markdown("""
<style>

/* Remove Streamlit default padding */
.block-container {
    padding-top: 0rem;
    padding-bottom: 0rem;
    padding-left: 0rem;
    padding-right: 0rem;
}

/* Remove top header space */
header {
    visibility: hidden;
    height: 0px;
}

/* Full black background */
.stApp {
    background-color: #141414;
    color: white;
}

/* Netflix red title */
.netflix-title {
    color: #E50914;
    font-size: 48px;
    font-weight: 800;
    text-align: center;
    margin-top: 40px;
    letter-spacing: 2px;
}

/* Cards */
.card {
    background-color: #1f1f1f;
    padding: 24px;
    border-radius: 14px;
    margin: 30px auto;
    max-width: 700px;
    box-shadow: 0 0 25px rgba(0,0,0,0.8);
}

/* Labels */
label {
    color: white !important;
}

</style>
""", unsafe_allow_html=True)


# ---------------- TITLE ----------------
st.markdown("<div class='netflix-title'>NETFLIX</div>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center;'>Customer Churn Prediction</h3>", unsafe_allow_html=True)
# ---------------- LOAD FILES ----------------
with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

scaler = joblib.load("scaler.pkl")
model = joblib.load("churn_model.pkl")

city_features = [c for c in feature_columns if c.startswith("City_")]

# ---------------- INPUT CARD ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📋 Customer Details")

age = st.number_input("Age", 0, 120, 25)
tenure = st.number_input("Tenure (Months)", 0, 72, 12)
monthly_charge = st.number_input("Monthly Charge", 0.0, 150.0, 50.0)
satisfaction = st.slider("Satisfaction Score", 1, 5, 3)

gender = st.selectbox("Gender", ["Male", "Female"])
under_30 = st.selectbox("Under 30?", ["Yes", "No"])

city = st.selectbox(
    "City (optional)",
    [""] + [c.replace("City_", "") for c in city_features]
)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- DATA PREP ----------------
input_data = {col: 0 for col in feature_columns}

input_data.update({
    "Age": age,
    "Tenure in Months": tenure,
    "Monthly Charge": monthly_charge,
    "Satisfaction Score": satisfaction,
    "Gender_Male": 1 if gender == "Male" else 0,
    "Under 30_Yes": 1 if under_30 == "Yes" else 0
})

if city:
    input_data[f"City_{city}"] = 1

input_df = pd.DataFrame([input_data])

numeric_cols = [c for c in scaler.feature_names_in_ if c in input_df.columns]
input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

# ---------------- PREDICTION ----------------
prediction = model.predict(input_df)
prob = model.predict_proba(input_df)[:, 1][0]

# ---------------- RESULT CARD ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🎯 Prediction Result")

if prediction[0] == 1:
    st.markdown(
        f"<h3 style='color:#E50914;'>⚠️ Likely to Churn</h3>"
        f"<p>Probability: {prob*100:.2f}%</p>",
        unsafe_allow_html=True
    )
else:
    st.markdown(
        f"<h3 style='color:#46d369;'>✅ Loyal Customer</h3>"
        f"<p>Probability: {prob*100:.2f}%</p>",
        unsafe_allow_html=True
    )

st.markdown("</div>", unsafe_allow_html=True)
