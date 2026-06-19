
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="CreditWise Loan Predictor", page_icon="🏦")

st.title("🏦 CreditWise Loan Approval Predictor")

@st.cache_resource
def load_artifacts():
    model = joblib.load("loan_model.pkl")
    scaler = joblib.load("scaler.pkl")
    ohe = joblib.load("encoder.pkl")
    return model, scaler, ohe

try:
    model, scaler, ohe = load_artifacts()
except Exception as e:
    st.error(f"Model files not found: {e}")
    st.stop()

st.header("Applicant Information")

education = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD"])
employment = st.selectbox("Employment Status", ["Employed", "Self-Employed", "Unemployed"])
marital = st.selectbox("Marital Status", ["Single", "Married"])
purpose = st.selectbox("Loan Purpose", ["Home", "Car", "Education", "Business", "Personal"])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
gender = st.selectbox("Gender", ["Male", "Female"])
employer_cat = st.selectbox("Employer Category", ["Government", "Private", "Business", "Other"])

app_income = st.number_input("Applicant Income", min_value=0.0)
co_income = st.number_input("Coapplicant Income", min_value=0.0)
credit_score = st.number_input("Credit Score", min_value=0.0)
dti_ratio = st.number_input("DTI Ratio", min_value=0.0)
savings = st.number_input("Savings", min_value=0.0)

if st.button("Predict Loan Approval"):
    try:
        data = pd.DataFrame({
            "Education_Level":[education],
            "Employment_Status":[employment],
            "Marital_Status":[marital],
            "Loan_Purpose":[purpose],
            "Property_Area":[property_area],
            "Gender":[gender],
            "Employer_Category":[employer_cat],
            "Applicant_Income":[app_income],
            "Coapplicant_Income":[co_income],
            "Savings":[savings],
            "DTI_Ratio_sq":[dti_ratio ** 2],
            "Credit_Score_sq":[credit_score ** 2]
        })

        data["Education_Level"] = data["Education_Level"].map({
            "High School":0,
            "Bachelor":1,
            "Master":2,
            "PhD":3
        }).fillna(0)

        cat_cols = ["Employment_Status","Marital_Status","Loan_Purpose",
                    "Property_Area","Gender","Employer_Category"]

        encoded = ohe.transform(data[cat_cols])
        encoded_df = pd.DataFrame(
            encoded,
            columns=ohe.get_feature_names_out(cat_cols)
        )

        final_df = pd.concat(
            [data.drop(columns=cat_cols).reset_index(drop=True),
             encoded_df.reset_index(drop=True)],
            axis=1
        )

        scaled = scaler.transform(final_df)

        prediction = model.predict(scaled)[0]

        if prediction == 1:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Rejected")

    except Exception as e:
        st.error(f"Prediction error: {e}")
