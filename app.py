import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from train_model import train_and_save

st.set_page_config(page_title="Employee Attrition Prediction", page_icon="📊", layout="wide")

DATA_PATH = Path("Employee_Attrition.csv")
MODEL_PATH = Path("attrition_model.joblib")

@st.cache_resource
def load_bundle():
    if not MODEL_PATH.exists():
        train_and_save()
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

st.title("📊 Employee Attrition Prediction")
st.caption("Machine Learning Mini Project | Predict whether an employee may stay or leave")

try:
    data = load_data()
    bundle = load_bundle()
except Exception as e:
    st.error(f"Unable to load the project files: {e}")
    st.stop()

st.info("Enter employee details below and select Predict Attrition. The model uses the Gradient Boosting classifier selected in the project.")
num_cols = bundle["num_cols"]
cat_cols = bundle["cat_cols"]
feature_cols = bundle["feature_columns"]

with st.form("employee_form"):
    st.subheader("Employee details")
    inputs = {}
    left, right = st.columns(2)

    for idx, col in enumerate(feature_cols):
        target = left if idx % 2 == 0 else right
        with target:
            if col in num_cols:
                low = int(data[col].min())
                high = int(data[col].max())
                default = int(data[col].median())
                inputs[col] = st.number_input(
                    col.replace("_", " "), min_value=low, max_value=high,
                    value=default, step=1
                )
            else:
                opts = sorted(data[col].dropna().astype(str).unique().tolist())
                inputs[col] = st.selectbox(col.replace("_", " "), opts)

    submitted = st.form_submit_button("🔍 Predict Attrition", use_container_width=True)

if submitted:
    row = pd.DataFrame([inputs], columns=feature_cols)
    transformed = bundle["preprocessor"].transform(row)
    prediction = int(bundle["model"].predict(transformed)[0])
    probability = float(bundle["model"].predict_proba(transformed)[0][1])

    st.divider()
    st.subheader("Prediction result")
    c1, c2 = st.columns(2)
    with c1:
        if prediction == 1:
            st.error("Prediction: Employee may leave")
        else:
            st.success("Prediction: Employee may stay")
    with c2:
        st.metric("Estimated attrition probability", f"{probability * 100:.1f}%")
        st.progress(probability)

    st.caption("This is a model estimate, not a certainty. Do not use it as the sole basis for employment decisions.")
    with st.expander("Show submitted employee details"):
        st.dataframe(row, use_container_width=True)

st.divider()
st.subheader("About this project")
a, b, c = st.columns(3)
a.metric("Dataset records", f"{len(data):,}")
b.metric("Original dataset columns", "35")
c.metric("Selected model", "Gradient Boosting")
st.write("The model uses numerical scaling, one-hot encoding, SMOTE on the training data, and Gradient Boosting classification.")
