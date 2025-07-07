import streamlit as st
import numpy as np
import joblib
import os

# Load model
BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "xgb_adr_model.pkl")
model = joblib.load(model_path)

st.set_page_config(page_title="ADR Risk Predictor 💊", layout="centered")

st.title("💊 ADR (Adverse Drug Reaction) Risk Predictor")

st.markdown("Please enter the following patient and medication details:")

# Input fields
age = st.number_input("Age", min_value=0, max_value=120, step=1)
gender = st.selectbox("Gender", ("Male", "Female"))
gender_val = 1 if gender == "Male" else 0

med_count = st.number_input("Number of Medications", min_value=0, max_value=50, step=1)

condition = st.radio("Does the patient have a chronic condition?", ("Yes", "No"))
condition_val = 1 if condition == "Yes" else 0

visits = st.number_input("Number of Doctor Visits in Last Year", min_value=0, max_value=50, step=1)

allergies = st.radio("Does the patient have any known allergies?", ("Yes", "No"))
allergies_val = 1 if allergies == "Yes" else 0

vaccines = st.radio("Is the patient vaccinated (relevant vaccines)?", ("Yes", "No"))
vaccines_val = 1 if vaccines == "Yes" else 0

# Predict
if st.button("Predict ADR Risk"):
    features = np.array([[med_count, med_count, condition_val, visits, 1, age, gender_val, allergies_val, vaccines_val]])
    prob = model.predict_proba(features)[0][1]
    if prob >= 0.3:
        st.error(f"⚠️ High ADR Risk: {prob:.2f}")
    else:
        st.success(f"✅ Low ADR Risk: {prob:.2f}")
