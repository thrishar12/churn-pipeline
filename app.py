import joblib, pandas as pd, streamlit as st

st.title("Telecom Churn")
df = pd.read_csv("data/telco.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
model = joblib.load("model.joblib")

tab1, tab2 = st.tabs(["EDA", "Predict"])
with tab1:
    st.subheader("Churn rate by contract")
    st.bar_chart(df.groupby("Contract")["Churn"].apply(lambda s: (s == "Yes").mean()))
    st.dataframe(df.describe())
with tab2:
    row = df.drop(columns=["customerID", "Churn"]).iloc[[0]].copy()
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly = st.number_input("Monthly charges", 18.0, 120.0, 70.0)
    contract = st.selectbox("Contract", df["Contract"].unique())
    internet = st.selectbox("Internet service", df["InternetService"].unique())
    if st.button("Predict"):
        row["tenure"], row["MonthlyCharges"], row["TotalCharges"] = tenure, monthly, tenure * monthly
        row["Contract"], row["InternetService"] = contract, internet
        st.metric("Churn probability", f"{model.predict_proba(row)[0, 1]:.0%}")