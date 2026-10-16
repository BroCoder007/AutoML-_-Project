"""Streamlit interface for gas classification from 128 sensor features."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
MODEL_PATH = PROJECT_ROOT / "models" / "gas_classifier.joblib"
FEATURE_COLUMNS = [f"Feature{i}" for i in range(1, 129)]


@st.cache_resource
def load_model_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model bundle not found at {MODEL_PATH}. "
            "Run `python src/train_app_model.py` from gas_sensor_ml_capstone first."
        )
    bundle = joblib.load(MODEL_PATH)
    missing = {"model", "feature_columns", "gas_names"} - set(bundle)
    if missing:
        raise ValueError(f"Model bundle is missing required keys: {sorted(missing)}")
    if bundle["feature_columns"] != FEATURE_COLUMNS:
        raise ValueError("The model feature order does not match Feature1 through Feature128.")
    return bundle


def predict(bundle, values):
    row = pd.DataFrame([values], columns=FEATURE_COLUMNS)
    predicted_class = int(bundle["model"].predict(row)[0])
    probabilities = bundle["model"].predict_proba(row)[0]
    classes = bundle["model"].named_steps["classifier"].classes_
    probability_table = (
        pd.DataFrame(
            {
                "Gas": [bundle["gas_names"].get(int(label), str(label)) for label in classes],
                "Probability": probabilities,
            }
        )
        .sort_values("Probability", ascending=False)
        .reset_index(drop=True)
    )
    return predicted_class, probability_table


st.set_page_config(page_title="Gas Sensor Classifier", page_icon="🧪", layout="wide")
st.title("Gas Sensor Array Classifier")
st.write(
    "Upload one row of sensor measurements or enter values manually to predict the gas class."
)
st.caption("The model uses 128 sensor features and a persisted Random Forest pipeline.")

try:
    bundle = load_model_bundle()
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.stop()

input_mode = st.radio("Input method", ["CSV upload", "Manual entry"], horizontal=True)
values = None

if input_mode == "CSV upload":
    uploaded_file = st.file_uploader(
        "Upload a CSV containing Feature1 through Feature128",
        type=["csv"],
    )
    if uploaded_file is not None:
        uploaded = pd.read_csv(uploaded_file)
        missing_columns = [column for column in FEATURE_COLUMNS if column not in uploaded]
        if missing_columns:
            st.error(f"CSV is missing {len(missing_columns)} required feature columns.")
        elif uploaded.empty:
            st.error("The uploaded CSV does not contain any rows.")
        else:
            values = uploaded[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").iloc[0]
            if values.isna().any():
                st.error("The first uploaded row contains missing or non-numeric feature values.")
                values = None
            else:
                st.info("Using the first row from the uploaded CSV.")
else:
    with st.expander("Enter the 128 sensor values", expanded=False):
        st.caption("Values default to 0.0. Replace them with one complete sensor measurement.")
        manual_values = []
        columns = st.columns(4)
        for index, feature in enumerate(FEATURE_COLUMNS):
            with columns[index % 4]:
                manual_values.append(
                    st.number_input(feature, value=0.0, format="%.6f", key=feature)
                )
        values = manual_values

if values is not None and st.button("Predict gas class", type="primary"):
    predicted_class, probability_table = predict(bundle, values)
    gas_name = bundle["gas_names"].get(predicted_class, str(predicted_class))
    st.success(f"Predicted gas: {gas_name} (class {predicted_class})")
    st.subheader("Class probabilities")
    st.bar_chart(probability_table.set_index("Gas"))
    st.dataframe(
        probability_table.style.format({"Probability": "{:.2%}"}),
        use_container_width=True,
        hide_index=True,
    )
