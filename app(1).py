import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
ohe = pickle.load(open("ohe.pkl", "rb"))

st.title("CreditWise Loan Approval Predictor")

applicant_income = st.number_input("Applicant Income", min_value=0.0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0)
age = st.number_input("Age", min_value=18)
dependents = st.number_input("Dependents", min_value=0)
existing_loans = st.number_input("Existing Loans", min_value=0)
savings = st.number_input("Savings", min_value=0.0)
collateral_value = st.number_input("Collateral Value", min_value=0.0)
loan_amount = st.number_input("Loan Amount", min_value=0.0)
loan_term = st.number_input("Loan Term", min_value=1)
credit_score = st.number_input("Credit Score", min_value=0.0)
dti_ratio = st.number_input("DTI Ratio", min_value=0.0)

employment_status = st.text_input("Employment Status")
marital_status = st.text_input("Marital Status")
loan_purpose = st.text_input("Loan Purpose")
property_area = st.text_input("Property Area")
education_level = st.number_input("Education Level Encoded", min_value=0)
gender = st.text_input("Gender")
employer_category = st.text_input("Employer Category")

if st.button("Predict"):
    input_df = pd.DataFrame({
        "Applicant_Income":[applicant_income],
        "Coapplicant_Income":[coapplicant_income],
        "Age":[age],
        "Dependents":[dependents],
        "Existing_Loans":[existing_loans],
        "Savings":[savings],
        "Collateral_Value":[collateral_value],
        "Loan_Amount":[loan_amount],
        "Loan_Term":[loan_term],
        "Education_Level":[education_level],
        "Credit_Score_sq":[credit_score**2],
        "DTI_Ratio_sq":[dti_ratio**2]
    })

    cat_df = pd.DataFrame({
        "Employment_Status":[employment_status],
        "Marital_Status":[marital_status],
        "Loan_Purpose":[loan_purpose],
        "Property_Area":[property_area],
        "Gender":[gender],
        "Employer_Category":[employer_category]
    })

    encoded = ohe.transform(cat_df)
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out())

    final_df = pd.concat([input_df, encoded_df], axis=1)

    scaled = scaler.transform(final_df)
    prediction = model.predict(scaled)[0]

    if prediction == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Rejected")
