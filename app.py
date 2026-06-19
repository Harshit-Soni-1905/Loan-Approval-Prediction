import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title="CreditWise Loan Predictor",
    page_icon="🏦",
    layout="wide"
)

# Load models
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
ohe = pickle.load(open("ohe.pkl", "rb"))

# Custom CSS
st.markdown("""
<style>
.main {
    padding-top: 1rem;
}

.stButton > button {
    width: 100%;
    height: 3.2em;
    font-size: 20px;
    border-radius: 12px;
}

.metric-card {
    background-color: #1f2937;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.title("🏦 CreditWise Loan Approval Predictor")
st.markdown("### AI Powered Loan Approval Assessment")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model", "Naive Bayes")

with col2:
    st.metric("Type", "Classification")

with col3:
    st.metric("Status", "Live")

st.divider()

st.subheader("📋 Applicant Information")

col1, col2 = st.columns(2)

with col1:
    applicant_income = st.number_input("💰 Applicant Income", min_value=0.0)
    age = st.number_input("🎂 Age", min_value=18)
    savings = st.number_input("🏦 Savings", min_value=0.0)
    loan_amount = st.number_input("💵 Loan Amount", min_value=0.0)
    credit_score = st.number_input("📊 Credit Score", min_value=0.0)
    education_level = st.selectbox(
        "🎓 Education Level",
        ["Graduate", "Not Graduate"]
    )

    marital_status = st.selectbox(
        "💍 Marital Status",
        ["Married", "Single"]
    )

    gender = st.selectbox(
        "👤 Gender",
        ["Female", "Male"]
    )

with col2:
    coapplicant_income = st.number_input("👥 Coapplicant Income", min_value=0.0)
    dependents = st.number_input("👨‍👩‍👧 Dependents", min_value=0)
    existing_loans = st.number_input("📁 Existing Loans", min_value=0)
    loan_term = st.number_input("📅 Loan Term", min_value=1)
    dti_ratio = st.number_input("📈 DTI Ratio", min_value=0.0)
    collateral_value = st.number_input("🏠 Collateral Value", min_value=0.0)

    employment_status = st.selectbox(
        "💼 Employment Status",
        ["Contract", "Salaried", "Self-employed", "Unemployed"]
    )

    loan_purpose = st.selectbox(
        "🎯 Loan Purpose",
        ["Business", "Car", "Education", "Home", "Personal"]
    )

    property_area = st.selectbox(
        "📍 Property Area",
        ["Rural", "Semiurban", "Urban"]
    )

    employer_category = st.selectbox(
        "🏢 Employer Category",
        ["Business", "Government", "MNC", "Private", "Unemployed"]
    )

st.divider()

if st.button("🚀 Predict Loan Approval"):

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

    st.divider()

    if prediction == 1:
        st.success("✅ LOAN APPROVED")
        st.balloons()
    else:
        st.error("❌ LOAN REJECTED")

st.markdown("---")
st.caption("Built with Streamlit • CreditWise Loan Predictor")
