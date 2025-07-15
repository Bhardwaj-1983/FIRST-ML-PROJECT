import streamlit as st
import joblib
import numpy as np

model = joblib.load("xgboost_best_model.pkl")

st.title("Income Prediction App")

age = st.slider("Age", 18, 90, 30)
capital_gain = st.number_input("Capital Gain", value=0)
capital_loss = st.number_input("Capital Loss", value=0)
hours_per_week = st.slider("Hours per Week", 1, 99, 40)

education_mapping = {
    "HS-grad": 9,
    "Some college": 10,
    "Associate's degree": 11,
    "Bachelor's degree": 13,
    "Master's degree": 14,
    "Doctorate": 16,
    "Prof-school": 15,
    "Other (e.g. 12th, 11th, 10th etc.)": 8
}

education_level = st.selectbox("Education Level", list(education_mapping.keys()))
education_num = education_mapping[education_level]

input_data = np.array([[age, capital_gain, hours_per_week, education_num, capital_loss]])

if st.button("Predict Income"):
    prediction = model.predict(input_data)[0]
    result = ">50K" if prediction == 1 else "<=50K"
    st.success(f"Predicted Income Category: {result}")
