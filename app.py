import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
ohe = pickle.load(open("ohe.pkl", "rb"))

st.title("🏦 CreditWise Loan Approval Predictor")

st.subheader("Enter Applicant Details")

# Numerical inputs
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

education_level = st.selectbox("Education Level", ["Graduate", "Not Graduate"])

employment_status = st.selectbox(
    "Employment Status",
    ["Contract", "Salaried", "Self-employed", "Unemployed"]
)

marital_status = st.selectbox(
    "Marital Status",
    ["Married", "Single"]
)

loan_purpose = st.selectbox(
    "Loan Purpose",
    ["Business", "Car", "Education", "Home", "Personal"]
)

property_area = st.selectbox(
    "Property Area",
    ["Rural", "Semiurban", "Urban"]
)

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

employer_category = st.selectbox(
    "Employer Category",
    ["Business", "Government", "MNC", "Private", "Unemployed"]
)

if st.button("Predict Loan Approval"):

    base_df = pd.DataFrame({
        "Applicant_Income": [applicant_income],
        "Coapplicant_Income": [coapplicant_income],
        "Age": [age],
        "Dependents": [dependents],
        "Existing_Loans": [existing_loans],
        "Savings": [savings],
        "Collateral_Value": [collateral_value],
        "Loan_Amount": [loan_amount],
        "Loan_Term": [loan_term],
        "Education_Level": [1 if education_level == "Graduate" else 0],
    })

    cat_df = pd.DataFrame({
        "Employment_Status": [employment_status],
        "Marital_Status": [marital_status],
        "Loan_Purpose": [loan_purpose],
        "Property_Area": [property_area],
        "Gender": [gender],
        "Employer_Category": [employer_category]
    })

    encoded = ohe.transform(cat_df)

    encoded_df = pd.DataFrame(
        encoded,
        columns=ohe.get_feature_names_out()
    )

    final_df = pd.concat([base_df, encoded_df], axis=1)

    final_df["DTI_Ratio_sq"] = dti_ratio ** 2
    final_df["Credit_Score_sq"] = credit_score ** 2

    final_df = final_df[scaler.feature_names_in_]

    scaled = scaler.transform(final_df)

    prediction = model.predict(scaled)[0]

    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")
