# -*- coding: utf-8 -*-
"""
CreditWise Loan Approval Prediction - Streamlit App
"""

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CreditWise – Loan Approval",
    page_icon="💳",
    layout="wide"
)

st.title("💳 CreditWise – Loan Approval Prediction")
st.markdown("Upload your `loan_approval_data.csv` to run EDA, train models, and predict loan approvals.")

# ── File upload ───────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload loan_approval_data.csv", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a CSV file to get started.")
    st.stop()

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_and_preprocess(file):
    df = pd.read_csv(file)

    # ── Handle Missing Values ─────────────────────────────────────────────────
    categorical_cols = df.select_dtypes(include=["object"]).columns
    numerical_cols   = df.select_dtypes(include=["number"]).columns

    num_imp = SimpleImputer(strategy="mean")
    df[numerical_cols] = num_imp.fit_transform(df[numerical_cols])

    cat_imp = SimpleImputer(strategy="most_frequent")
    df[categorical_cols] = cat_imp.fit_transform(df[categorical_cols])

    # ── Drop ID column if present ─────────────────────────────────────────────
    if "Applicant_ID" in df.columns:
        df = df.drop("Applicant_ID", axis=1)

    # ── Encoding ──────────────────────────────────────────────────────────────
    le = LabelEncoder()
    if "Education_Level" in df.columns:
        df["Education_Level"] = le.fit_transform(df["Education_Level"])
    if "Loan_Approved" in df.columns:
        df["Loan_Approved"] = le.fit_transform(df["Loan_Approved"])

    ohe_cols = [
        c for c in ["Employment_Status", "Marital_Status", "Loan_Purpose",
                     "Property_Area", "Gender", "Employer_Category"]
        if c in df.columns
    ]
    if ohe_cols:
        ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
        encoded    = ohe.fit_transform(df[ohe_cols])
        encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(ohe_cols), index=df.index)
        df = pd.concat([df.drop(columns=ohe_cols), encoded_df], axis=1)

    return df

df = load_and_preprocess(uploaded_file)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA", "🔥 Correlation", "🤖 Model Training", "🔮 Predict"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 – EDA
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Exploratory Data Analysis")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    st.write(f"Shape: **{df.shape[0]} rows × {df.shape[1]} columns**")

    col1, col2 = st.columns(2)

    # ── Pie chart – class balance ─────────────────────────────────────────
    with col1:
        st.subheader("Loan Approval Balance")
        if "Loan_Approved" in df.columns:
            fig, ax = plt.subplots()
            counts = df["Loan_Approved"].value_counts()
            ax.pie(counts, labels=["No", "Yes"], autopct="%1.1f%%", startangle=90)
            ax.set_title("Is Loan Approved?")
            st.pyplot(fig)
            plt.close(fig)

    # ── Education Level bar chart ─────────────────────────────────────────
    with col2:
        st.subheader("Education Level Distribution")
        if "Education_Level" in df.columns:
            fig, ax = plt.subplots()
            edu_counts = df["Education_Level"].value_counts()
            sns.barplot(x=edu_counts.index, y=edu_counts.values, ax=ax)
            ax.set_xlabel("Education Level (encoded)")
            ax.set_ylabel("Count")
            for container in ax.containers:
                ax.bar_label(container)
            st.pyplot(fig)
            plt.close(fig)

    # ── Income distributions ──────────────────────────────────────────────
    st.subheader("Income Distributions")
    col3, col4 = st.columns(2)

    with col3:
        if "Applicant_Income" in df.columns:
            fig, ax = plt.subplots()
            sns.histplot(data=df, x="Applicant_Income", bins=20, ax=ax)
            ax.set_title("Applicant Income")
            st.pyplot(fig)
            plt.close(fig)

    with col4:
        if "Coapplicant_Income" in df.columns:
            fig, ax = plt.subplots()
            sns.histplot(data=df, x="Coapplicant_Income", bins=20, ax=ax)
            ax.set_title("Co-applicant Income")
            st.pyplot(fig)
            plt.close(fig)

    # ── Box plots ─────────────────────────────────────────────────────────
    st.subheader("Box Plots vs Loan Approval")
    box_cols = [c for c in ["Applicant_Income", "Credit_Score", "DTI_Ratio", "Savings"] if c in df.columns]

    if box_cols and "Loan_Approved" in df.columns:
        ncols = 2
        nrows = (len(box_cols) + 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows))
        axes = np.array(axes).flatten()
        for i, col in enumerate(box_cols):
            sns.boxplot(ax=axes[i], data=df, x="Loan_Approved", y=col)
            axes[i].set_title(col)
        # hide unused axes
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Histplots with hue ────────────────────────────────────────────────
    st.subheader("Feature Distribution by Loan Approval")
    col5, col6 = st.columns(2)

    with col5:
        if "Credit_Score" in df.columns and "Loan_Approved" in df.columns:
            fig, ax = plt.subplots()
            sns.histplot(data=df, x="Credit_Score", hue="Loan_Approved", bins=20, multiple="dodge", ax=ax)
            ax.set_title("Credit Score by Approval")
            st.pyplot(fig)
            plt.close(fig)

    with col6:
        if "Applicant_Income" in df.columns and "Loan_Approved" in df.columns:
            fig, ax = plt.subplots()
            sns.histplot(data=df, x="Applicant_Income", hue="Loan_Approved", bins=20, multiple="dodge", ax=ax)
            ax.set_title("Applicant Income by Approval")
            st.pyplot(fig)
            plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 – Correlation
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Correlation Heatmap")

    num_cols = df.select_dtypes(include="number")
    corr_matrix = num_cols.corr()

    fig, ax = plt.subplots(figsize=(15, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)
    plt.close(fig)

    if "Loan_Approved" in df.columns:
        st.subheader("Correlation with Loan Approval")
        corr_series = num_cols.corr()["Loan_Approved"].sort_values(ascending=False)
        st.dataframe(corr_series.rename("Correlation").reset_index(), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 – Model Training
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Model Training & Evaluation")

    @st.cache_data
    def train_models(df):
        # ── Feature Engineering ───────────────────────────────────────────
        df = df.copy()
        df["DTI_Ratio_sq"]    = df["DTI_Ratio"] ** 2    if "DTI_Ratio"    in df.columns else 0
        df["Credit_Score_sq"] = df["Credit_Score"] ** 2 if "Credit_Score" in df.columns else 0

        drop_cols = [c for c in ["Loan_Approved", "Credit_Score", "DTI_Ratio"] if c in df.columns]
        X = df.drop(columns=drop_cols)
        y = df["Loan_Approved"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

        def evaluate(name, model, X_tr, X_te, y_tr, y_te):
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            return {
                "Model":     name,
                "Precision": round(precision_score(y_te, y_pred), 4),
                "Recall":    round(recall_score(y_te, y_pred), 4),
                "F1 Score":  round(f1_score(y_te, y_pred), 4),
                "Accuracy":  round(accuracy_score(y_te, y_pred), 4),
                "CM":        confusion_matrix(y_te, y_pred),
            }

        results = [
            evaluate("Logistic Regression", LogisticRegression(max_iter=1000),
                     X_train_scaled, X_test_scaled, y_train, y_test),
            evaluate("KNN (k=7)", KNeighborsClassifier(n_neighbors=7),
                     X_train_scaled, X_test_scaled, y_train, y_test),
            evaluate("Naive Bayes", GaussianNB(),
                     X_train_scaled, X_test_scaled, y_train, y_test),
        ]
        return results, scaler, X.columns.tolist()

    results, scaler, feature_cols = train_models(df)

    # ── Metrics table ─────────────────────────────────────────────────────
    metrics_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != "CM"} for r in results
    ])
    st.subheader("Model Comparison")
    st.dataframe(metrics_df.set_index("Model"), use_container_width=True)

    best = metrics_df.loc[metrics_df["Precision"].idxmax(), "Model"]
    st.success(f"✅ Best model by Precision: **{best}**")

    # ── Confusion matrices ────────────────────────────────────────────────
    st.subheader("Confusion Matrices")
    cols = st.columns(len(results))
    for col, res in zip(cols, results):
        with col:
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(res["CM"], annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_title(res["Model"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
            plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 – Predict
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Predict Loan Approval")
    st.markdown("Enter applicant details below to get a real-time prediction.")

    @st.cache_resource
    def get_best_model(df):
        """Train the best model (Naive Bayes) on full data for inference."""
        df = df.copy()
        df["DTI_Ratio_sq"]    = df["DTI_Ratio"] ** 2    if "DTI_Ratio"    in df.columns else 0
        df["Credit_Score_sq"] = df["Credit_Score"] ** 2 if "Credit_Score" in df.columns else 0

        drop_cols = [c for c in ["Loan_Approved", "Credit_Score", "DTI_Ratio"] if c in df.columns]
        X = df.drop(columns=drop_cols)
        y = df["Loan_Approved"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = GaussianNB()
        model.fit(X_scaled, y)
        return model, scaler, X.columns.tolist()

    inf_model, inf_scaler, inf_cols = get_best_model(df)

    # ── Build input form dynamically from numeric columns ─────────────────
    numeric_input_cols = [c for c in inf_cols if df[c].nunique() > 2][:8]  # top 8 numeric
    binary_cols        = [c for c in inf_cols if c not in numeric_input_cols]

    with st.form("predict_form"):
        st.subheader("Applicant Details")
        user_input = {}

        grid = st.columns(3)
        for i, col in enumerate(numeric_input_cols):
            col_min = float(df[col].min())
            col_max = float(df[col].max())
            col_mean = float(df[col].mean())
            user_input[col] = grid[i % 3].number_input(
                col.replace("_", " "), min_value=col_min, max_value=col_max, value=col_mean
            )

        # binary / one-hot cols default to 0
        for col in binary_cols:
            user_input[col] = 0.0

        submitted = st.form_submit_button("Predict", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame([user_input])[inf_cols]

        # Compute engineered features
        if "DTI_Ratio" in input_df.columns:
            input_df["DTI_Ratio_sq"] = input_df["DTI_Ratio"] ** 2
        if "Credit_Score" in input_df.columns:
            input_df["Credit_Score_sq"] = input_df["Credit_Score"] ** 2

        # Align columns
        input_df = input_df.reindex(columns=inf_cols, fill_value=0)
        input_scaled = inf_scaler.transform(input_df)
        prediction   = inf_model.predict(input_scaled)[0]
        proba        = inf_model.predict_proba(input_scaled)[0]

        st.markdown("---")
        if prediction == 1:
            st.success(f"✅ **Loan Approved** — confidence: {proba[1]*100:.1f}%")
        else:
            st.error(f"❌ **Loan Rejected** — confidence: {proba[0]*100:.1f}%")

        col_a, col_b = st.columns(2)
        col_a.metric("Approval Probability",  f"{proba[1]*100:.1f}%")
        col_b.metric("Rejection Probability", f"{proba[0]*100:.1f}%")
