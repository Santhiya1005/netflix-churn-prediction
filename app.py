import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from pathlib import Path
import re

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Netflix Churn Retention Dashboard",
    page_icon="🎬",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
header[data-testid="stHeader"] {display: none;}
div[data-testid="stToolbar"] {display: none;}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

.stApp {
    background: #0b0b0b;
    color: #ffffff;
    font-family: "Segoe UI", sans-serif;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 1.5rem;
    padding-left: 2rem;
    padding-right: 2rem;
    max-width: 1400px;
}

section[data-testid="stSidebar"] {
    background: #111111 !important;
    border-right: 1px solid rgba(255,255,255,0.08);
}
section[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

.main-title {
    font-size: 2.6rem;
    font-weight: 800;
    color: #E50914;
    margin-bottom: 0.3rem;
}
.sub-text {
    color: #E5E5E5;
    font-size: 1rem;
    margin-bottom: 1.4rem;
}
.badge {
    display: inline-block;
    padding: 0.35rem 0.8rem;
    border-radius: 999px;
    background: rgba(229, 9, 20, 0.12);
    color: #ff4d57;
    border: 1px solid rgba(229, 9, 20, 0.22);
    font-size: 0.82rem;
    font-weight: 700;
    margin-bottom: 0.8rem;
}

.card {
    background: #141414;
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 16px;
    padding: 1.1rem 1.2rem;
    margin-bottom: 1rem;
}
.card-title {
    color: #FFFFFF;
    font-size: 1.08rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}
.soft-text {
    color: #F2F2F2;
    font-size: 0.98rem;
    line-height: 1.6;
}

div[data-testid="metric-container"] {
    background: #161616 !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 16px !important;
    padding: 1rem !important;
}
div[data-testid="metric-container"] * {
    color: #ffffff !important;
    opacity: 1 !important;
}
div[data-testid="metric-container"] label,
div[data-testid="metric-container"] [data-testid="stMetricLabel"],
div[data-testid="metric-container"] p {
    color: #f2f2f2 !important;
    opacity: 1 !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #ffffff !important;
    opacity: 1 !important;
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    line-height: 1.2 !important;
}
div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    color: #ffffff !important;
    opacity: 1 !important;
}

.stSelectbox label,
.stNumberInput label,
.stSlider label,
.stTextInput label {
    color: #F5F5F5 !important;
    opacity: 1 !important;
    font-weight: 600 !important;
}
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: #181818 !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important;
    color: #FFFFFF !important;
}
div[data-baseweb="select"] span,
div[data-baseweb="input"] input {
    color: #FFFFFF !important;
    opacity: 1 !important;
}
button[kind="secondary"] {
    color: #FFFFFF !important;
}
.stSlider span {
    color: #FFFFFF !important;
    opacity: 1 !important;
}

.stButton > button,
.stFormSubmitButton > button {
    background: #E50914;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 1.2rem;
    font-weight: 700;
}
.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: #b20710;
    color: white;
}

[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.08);
}

[data-testid="stAlert"] {
    border-radius: 12px;
}
[data-testid="stMarkdownContainer"] {
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- PATHS ----------------
MODEL_PATH = "churn_model.pkl"
SCALER_PATH = "scaler.pkl"
FEATURES_PATH = "feature_columns.pkl"
DATA_PATH = "churn.csv"

# ---------------- LOADERS ----------------
@st.cache_resource
def load_model():
    if Path(MODEL_PATH).exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            with open(MODEL_PATH, "rb") as f:
                return pickle.load(f)
    return None

@st.cache_resource
def load_scaler():
    if Path(SCALER_PATH).exists():
        try:
            return joblib.load(SCALER_PATH)
        except Exception:
            with open(SCALER_PATH, "rb") as f:
                return pickle.load(f)
    return None

@st.cache_resource
def load_feature_columns():
    if Path(FEATURES_PATH).exists():
        try:
            cols = joblib.load(FEATURES_PATH)
        except Exception:
            with open(FEATURES_PATH, "rb") as f:
                cols = pickle.load(f)

        if isinstance(cols, pd.Index):
            cols = cols.tolist()
        elif isinstance(cols, np.ndarray):
            cols = cols.tolist()

        return list(cols)
    return []

@st.cache_data
def load_data():
    if Path(DATA_PATH).exists():
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame()

model = load_model()
scaler = load_scaler()
feature_columns = load_feature_columns()
df = load_data()

# ---------------- HELPERS ----------------
def normalize_name(text):
    return re.sub(r'[^a-z0-9]', '', str(text).strip().lower())

def find_col(df, possible_names):
    normalized_cols = {normalize_name(col): col for col in df.columns}
    for name in possible_names:
        key = normalize_name(name)
        if key in normalized_cols:
            return normalized_cols[key]
    return None

def churn_numeric(series):
    s = series.astype(str).str.strip().str.lower()
    return s.map({
        "yes": 1, "no": 0,
        "true": 1, "false": 0,
        "1": 1, "0": 0,
        "churned": 1, "stayed": 0
    })

def fill_if_exists(input_df, col_name, value):
    if col_name in input_df.columns:
        input_df[col_name] = value

def set_one_hot(input_df, prefixes, selected_value):
    selected_value = str(selected_value).strip().lower()
    if isinstance(prefixes, str):
        prefixes = [prefixes]

    for prefix in prefixes:
        prefix_lower = prefix.lower().strip() + "_"
        for col in input_df.columns:
            col_lower = col.lower().strip()
            if col_lower.startswith(prefix_lower):
                value_part = col.split("_", 1)[1].strip().lower()
                input_df[col] = 1 if value_part == selected_value else 0

def styled_card(title, text):
    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">{title}</div>
            <div class="soft-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
def metric_card(label, value):
    st.markdown(
        f"""
        <div style="
            background:#161616;
            border:1px solid rgba(255,255,255,0.10);
            border-radius:16px;
            padding:18px 20px;
            min-height:110px;
        ">
            <div style="
                color:#d9d9d9;
                font-size:16px;
                font-weight:600;
                margin-bottom:10px;
                opacity:1;
            ">{label}</div>
            <div style="
                color:#ffffff;
                font-size:42px;
                font-weight:800;
                line-height:1.1;
                opacity:1;
            ">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
def risk_label(prob):
    if prob >= 0.75:
        return "High Risk 🔴"
    elif prob >= 0.45:
        return "Medium Risk 🟠"
    return "Low Risk 🟢"

def retention_action(prob, contract, tenure, monthly_charges):
    suggestions = []

    if prob >= 0.75:
        suggestions.append("Initiate an immediate retention call.")
        suggestions.append("Provide a special discount or loyalty benefit.")
    elif prob >= 0.45:
        suggestions.append("Send personalized content recommendations.")
        suggestions.append("Launch a targeted engagement email or notification campaign.")
    else:
        suggestions.append("The customer is currently stable. Maintain regular engagement to retain them.")

    if contract == "Month-to-month":
        suggestions.append("Offer an upgrade to a yearly or long-term subscription plan.")
    if tenure < 12:
        suggestions.append("Provide onboarding support or welcome benefits to improve early customer experience.")
    if monthly_charges > 80:
        suggestions.append("Recommend a more cost-effective bundled or lower-priced plan.")

    return suggestions

def get_expected_columns():
    """
    Prefer scaler's expected columns.
    If scaler doesn't expose them, use model's feature_names_in_.
    Else fall back to feature_columns.pkl.
    """
    if scaler is not None and hasattr(scaler, "feature_names_in_"):
        return list(scaler.feature_names_in_)
    if model is not None and hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return list(feature_columns)

def build_input_from_form(
    gender, age, tenure, monthly_charges, total_charges,
    contract, payment_method, internet_service, paperless_billing
):
    expected_columns = get_expected_columns()
    input_df = pd.DataFrame([np.zeros(len(expected_columns))], columns=expected_columns)

    under_30 = 1 if age < 30 else 0
    senior_citizen = 1 if age >= 60 else 0
    married = 0
    dependents = 0
    number_of_dependents = 0

    direct_map = {
        "Age": age,
        "age": age,

        "Under 30": under_30,
        "Under30": under_30,

        "Senior Citizen": senior_citizen,
        "SeniorCitizen": senior_citizen,

        "Married": married,
        "Dependents": dependents,
        "Number of Dependents": number_of_dependents,
        "NumberOfDependents": number_of_dependents,

        "Tenure": tenure,
        "tenure": tenure,

        "Monthly Charges": monthly_charges,
        "MonthlyCharges": monthly_charges,
        "Monthly Charge": monthly_charges,

        "Total Charges": total_charges,
        "TotalCharges": total_charges,
        "Total Charge": total_charges,

        "Paperless Billing": int(paperless_billing),
        "PaperlessBilling": int(paperless_billing)
    }

    for col_name, value in direct_map.items():
        fill_if_exists(input_df, col_name, value)

    if "Gender" in input_df.columns:
        input_df["Gender"] = 1 if gender == "Male" else 0
    if "gender" in input_df.columns:
        input_df["gender"] = 1 if gender == "Male" else 0

    set_one_hot(input_df, ["Gender", "gender"], gender)
    set_one_hot(input_df, ["Contract", "contract"], contract)
    set_one_hot(input_df, ["PaymentMethod", "Payment Method", "payment_method"], payment_method)
    set_one_hot(input_df, ["InternetService", "Internet Service", "internet_service"], internet_service)
    set_one_hot(input_df, ["PaperlessBilling", "Paperless Billing"], str(paperless_billing))

    return input_df

def predict_customer(input_df):
    """
    Robust prediction pipeline:
    1. Align to scaler expected columns if available
    2. Scale if scaler exists
    3. Align to model expected columns if available
    4. Predict probability
    """
    X = input_df.copy()

    # Step 1: scale if possible
    if scaler is not None:
        if hasattr(scaler, "feature_names_in_"):
            scaler_cols = list(scaler.feature_names_in_)
            X_for_scaler = X.reindex(columns=scaler_cols, fill_value=0)
        else:
            X_for_scaler = X

        try:
            scaled = scaler.transform(X_for_scaler)
            if hasattr(scaler, "feature_names_in_"):
                X_scaled_df = pd.DataFrame(scaled, columns=list(scaler.feature_names_in_))
            else:
                X_scaled_df = pd.DataFrame(scaled)
        except Exception:
            # fallback without scaling
            X_scaled_df = X
    else:
        X_scaled_df = X

    # Step 2: align to model input
    if model is not None and hasattr(model, "feature_names_in_"):
        model_cols = list(model.feature_names_in_)
        X_for_model = X_scaled_df.reindex(columns=model_cols, fill_value=0)
    else:
        X_for_model = X_scaled_df

    # Step 3: probability
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(X_for_model)[0][1])
    else:
        pred = model.predict(X_for_model)[0]
        prob = float(pred)

    pred_label = 1 if prob >= 0.5 else 0
    return prob, pred_label, X_for_model

# ---------------- TITLE ----------------
st.markdown('<div class="badge">NETFLIX ANALYTICS</div>', unsafe_allow_html=True)
st.markdown('<div class="main-title">Netflix Subscription Retention Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-text">Role-based churn prediction dashboard with business insights and customer retention actions.</div>',
    unsafe_allow_html=True
)

# ---------------- SIDEBAR ----------------
role = st.sidebar.selectbox("Choose View", ["Admin", "Employee"])
st.sidebar.markdown("---")
st.sidebar.write("**Artifacts loaded:**")
st.sidebar.write(f"Model: {'✅' if model is not None else '❌'}")
st.sidebar.write(f"Scaler: {'✅' if scaler is not None else '❌'}")
st.sidebar.write(f"Features: {'✅' if len(feature_columns) > 0 else '❌'}")
st.sidebar.write(f"Dataset: {'✅' if not df.empty else '❌'}")

# ---------------- ADMIN VIEW ----------------
if role == "Admin":
    st.subheader("📊 Admin Dashboard")

    if df.empty:
        st.error("`churn.csv` could not be loaded.")
        st.stop()

    churn_col = find_col(df, ["Churn", "Churn Label", "Customer Status"])
    monthly_col = find_col(df, ["MonthlyCharges", "Monthly Charges", "Monthly Charge"])
    age_col = find_col(df, ["Age"])
    contract_col = find_col(df, ["Contract"])

    total_customers = len(df)

    if churn_col:
        churn_series = churn_numeric(df[churn_col]).fillna(0)
        churn_count = int(churn_series.sum())
        retained_count = int(total_customers - churn_count)
        churn_rate = round((churn_count / total_customers) * 100, 2)
    else:
        churn_count = 0
        retained_count = total_customers
        churn_rate = 0.0

    avg_monthly = round(df[monthly_col].mean(), 2) if monthly_col else 0.0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Total Customers", f"{total_customers}")
    with c2:
        metric_card("Churned Customers", f"{churn_count}")
    with c3:
        metric_card("Retained Customers", f"{retained_count}")
    with c4:
        metric_card("Avg Monthly Charge", f"${avg_monthly:.2f}")

    st.markdown("### 📈 Business Overview")

    info1, info2 = st.columns(2)
    with info1:
        styled_card(
            "Platform Churn Snapshot",
            "Monitor how many customers are leaving and which customer segments need retention focus."
        )
    with info2:
        styled_card(
            "Segment Insights",
            "Analyze customer behavior across age and pricing bands to identify churn-prone groups."
        )

    chart1, chart2 = st.columns(2)

    with chart1:
        if churn_col:
            st.write("#### Churn Distribution")
            churn_chart = df[churn_col].astype(str).value_counts()
            st.bar_chart(churn_chart, use_container_width=True)

        if contract_col and churn_col:
            st.write("#### Contract vs Churn")
            contract_churn = pd.crosstab(df[contract_col], df[churn_col])
            st.bar_chart(contract_churn, use_container_width=True)

    with chart2:
        if age_col and churn_col:
            st.write("#### Age Group vs Churn")
            age_df = df[[age_col, churn_col]].copy()
            age_df["Age Group"] = pd.cut(
                age_df[age_col],
                bins=[0, 30, 45, 60, 100],
                labels=["Under 30", "30-45", "46-60", "60+"]
            )
            age_group_chart = pd.crosstab(age_df["Age Group"], age_df[churn_col])
            st.bar_chart(age_group_chart, use_container_width=True)

        if monthly_col and churn_col:
            st.write("#### Monthly Charges by Churn")
            charge_df = df[[monthly_col, churn_col]].copy()
            charge_df["Charge Band"] = pd.cut(
                charge_df[monthly_col],
                bins=[0, 35, 70, 1000],
                labels=["Low", "Medium", "High"]
            )
            charge_chart = pd.crosstab(charge_df["Charge Band"], charge_df[churn_col])
            st.bar_chart(charge_chart, use_container_width=True)

    st.markdown("### 🧠 Key Insights")
    insights = []

    if contract_col and churn_col:
        temp = pd.crosstab(df[contract_col], churn_numeric(df[churn_col]), normalize="index") * 100
        if 1 in temp.columns:
            high_contract = temp[1].idxmax()
            high_contract_rate = round(temp[1].max(), 2)
            insights.append(f"{high_contract} customers show the highest churn rate at {high_contract_rate}%.")

    if age_col and churn_col:
        temp = df[[age_col, churn_col]].copy()
        temp["Age Group"] = pd.cut(
            temp[age_col],
            bins=[0, 30, 45, 60, 100],
            labels=["Under 30", "30-45", "46-60", "60+"]
        )
        grp = temp.groupby("Age Group", observed=False)[churn_col].apply(lambda x: churn_numeric(x).mean() * 100)
        if len(grp) > 0:
            insights.append(f"The highest age-group churn is observed in {grp.idxmax()} at {round(grp.max(), 2)}%.")

    if monthly_col:
        insights.append(f"The average monthly charge across customers is ${avg_monthly:.2f}.")
    if churn_col:
        insights.append(f"The overall churn rate is currently {churn_rate}%.")

    if not insights:
        insights.append("No insights could be generated. Please verify dataset column names.")

    for item in insights:
        styled_card("Insight", item)

    st.markdown("### 🔍 Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

# ---------------- EMPLOYEE VIEW ----------------
elif role == "Employee":
    st.subheader("👨‍💼 Employee / Analyst Dashboard")

    if model is None:
        st.error("`churn_model.pkl` could not be loaded.")
        st.stop()

    styled_card(
        "Single Customer Risk Prediction",
        "Enter a few core customer details to estimate churn probability and recommended retention action."
    )

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.slider("Age", 18, 80, 30)
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, value=6)

        with col2:
            monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=500.0, value=85.0, step=0.5)
            total_charges = st.number_input(
                "Total Charges",
                min_value=0.0,
                max_value=50000.0,
                value=float(max(tenure * monthly_charges, 0)),
                step=1.0
            )
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

        with col3:
            payment_method = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
            )
            internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            paperless_billing = st.selectbox("Paperless Billing", [True, False], index=0)

        submitted = st.form_submit_button("Predict Churn")

    if submitted:
        input_df = build_input_from_form(
            gender, age, tenure, monthly_charges, total_charges,
            contract, payment_method, internet_service, paperless_billing
        )

        try:
            probability, prediction, model_input = predict_customer(input_df)
            prediction_failed = False
        except Exception:
            probability = 0.0
            prediction = 0
            model_input = input_df.copy()
            prediction_failed = True

        risk = risk_label(probability)
        suggestions = retention_action(probability, contract, tenure, monthly_charges)

        if prediction_failed:
            st.warning(
                "Prediction could not be completed with the current model artifacts. "
                "The model, scaler, and saved feature set are likely from different training versions."
            )

        st.markdown("## 📌 Prediction Result")
        a, b, c = st.columns(3)
        with a:
            metric_card("Prediction", "Likely to Churn" if prediction == 1 else "Likely to Stay")
        with b:
            metric_card("Risk Score", f"{probability * 100:.2f}%")
        with c:
            metric_card("Risk Level", risk)

        if prediction == 1:
            st.error("The customer has a high probability of churn.")
        else:
            st.success("The customer is likely to stay.")

        st.markdown("### 🎯 Recommended Retention Actions")
        for s in suggestions:
            styled_card("Recommended Action", s)

        st.markdown("### 🧾 Input Summary")
        summary_df = pd.DataFrame({
            "Field": [
                "Gender", "Age", "Tenure", "Monthly Charges", "Total Charges",
                "Contract", "Payment Method", "Internet Service", "Paperless Billing"
            ],
            "Value": [
                gender, age, tenure, monthly_charges, total_charges,
                contract, payment_method, internet_service, paperless_billing
            ]
        })
        st.dataframe(summary_df, use_container_width=True)

        with st.expander("Show model input details"):
            st.write("Expected columns used for prediction:")
            st.write(list(model_input.columns))
            non_zero = [col for col in model_input.columns if model_input.iloc[0][col] != 0]
            st.write("Non-zero features:", non_zero)
            st.write(model_input)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built with Streamlit • Netflix-inspired churn retention dashboard")