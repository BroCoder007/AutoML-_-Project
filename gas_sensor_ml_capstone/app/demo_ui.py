"""Interactive regression, clustering, and classification demonstrations."""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from demo_models import SENSOR_COLUMNS, WEATHER_COLUMNS, assign_product, weather_features


WEATHER_LABELS = {
    "DEWP": "Dew point (C)", "TEMP": "Temperature (C)",
    "PRES": "Pressure (hPa)", "Iws": "Cumulative wind speed (m/s)",
    "Is": "Accumulated snow hours", "Ir": "Accumulated rain hours",
}


@st.cache_resource
def load_bundle(filename):
    path = PROJECT_ROOT / "models" / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {filename}. Run python gas_sensor_ml_capstone/src/train_demo_models.py "
            "from the repository root and commit the generated model files. "
            "For gas_classifier.joblib, run src/train_app_model.py from gas_sensor_ml_capstone."
        )
    return joblib.load(path)


@st.cache_data
def load_sensor_examples():
    return pd.read_csv(PROJECT_ROOT / "models/classification_examples.csv")


def render_regression():
    st.header("Predict PM2.5 from the weather")
    st.write("Choose a recorded example or enter six weather readings. The app calculates the extra temperature/dew-point feature for you.")
    bundle = load_bundle("regression_demo.joblib")
    model_name = st.selectbox("Regression model", list(bundle["models"]), key="regression_model")
    mode = st.radio("Weather input", ["Held-out example", "Edit weather values"], horizontal=True, key="weather_mode")
    examples = bundle["examples"]
    example_index = st.selectbox(
        "Weather example", list(range(len(examples))),
        format_func=lambda i: f"Example {i + 1} - {examples.iloc[i]['Observed at']}",
        key="weather_example",
    )
    example = examples.iloc[example_index]
    with st.form("weather_prediction"):
        if mode == "Edit weather values":
            columns = st.columns(3)
            values = {}
            for i, feature in enumerate(WEATHER_COLUMNS):
                with columns[i % 3]:
                    options = {"min_value": 0.0} if feature in ["Iws", "Is", "Ir"] else {}
                    if feature == "PRES":
                        options["min_value"] = 1.0
                    values[feature] = st.number_input(
                        WEATHER_LABELS[feature], value=float(example[feature]),
                        key=f"weather_{feature}_{example_index}", **options,
                    )
        else:
            values = {feature: float(example[feature]) for feature in WEATHER_COLUMNS}
            st.dataframe(
                pd.DataFrame({"Weather measurement": [WEATHER_LABELS[f] for f in WEATHER_COLUMNS],
                              "Value": [values[f] for f in WEATHER_COLUMNS]}),
                hide_index=True, width="stretch",
            )
        submitted = st.form_submit_button("Predict PM2.5", type="primary")
    if submitted:
        predicted = float(bundle["models"][model_name].predict(weather_features(pd.DataFrame([values])))[0])
        st.success(f"Predicted PM2.5: {predicted:.1f} ug/m3")
        columns = st.columns(3)
        columns[0].metric("Predicted PM2.5", f"{predicted:.1f} ug/m3")
        if mode == "Held-out example":
            actual = float(example["Actual PM2.5"])
            columns[1].metric("Recorded PM2.5", f"{actual:.1f} ug/m3")
            columns[2].metric("Absolute error", f"{abs(predicted - actual):.1f} ug/m3")
            st.caption("This example was held out of training for both regression models.")
        else:
            st.caption("Edited readings have no known target. Recorded PM2.5 is only shown for unchanged held-out examples.")
    st.divider()
    st.subheader("How the models performed")
    scores = bundle["metrics"].set_index("Model").loc[model_name]
    columns = st.columns(3)
    columns[0].metric("Test R2", f"{scores['R2']:.3f}")
    columns[1].metric("Test MAE", f"{scores['MAE']:.1f} ug/m3")
    columns[2].metric("Test RMSE", f"{scores['RMSE']:.1f} ug/m3")
    st.caption(
        f"Beijing hourly data, 2010-2014. Random 80/20 split: {bundle['training_rows']:,} training and "
        f"{bundle['test_rows']:,} test rows. This estimates historical readings; it is not a future forecast. "
        "These are retrained demo models, not the notebook's tuned model exports."
    )
    with st.expander("Compare both models and their held-out predictions"):
        st.dataframe(bundle["metrics"].round(3), hide_index=True, width="stretch")
        comparison = bundle["comparison"].rename(columns={model_name: "Predicted PM2.5"})
        st.scatter_chart(comparison, x="Actual PM2.5", y="Predicted PM2.5")
        st.caption("200 held-out readings. Points closer to equal actual and predicted values are better.")


def render_clustering():
    st.header("Discover groups of similar product titles")
    st.write("Enter a title to find its K-Means group, or explore the groups formed by hierarchical clustering.")
    bundle = load_bundle("clustering_demo.joblib")
    algorithm = st.radio(
        "Clustering algorithm", ["K-Means", "Agglomerative hierarchical"],
        horizontal=True, key="clustering_algorithm",
    )
    if algorithm == "K-Means":
        catalog = bundle["catalog"]
        summary = bundle["kmeans_summary"]
        silhouette = bundle["kmeans_silhouette"]
        with st.form("product_assignment"):
            title = st.text_input("Product title", value="apple iphone 8 plus 64gb silver", key="product_title")
            submitted = st.form_submit_button("Find product group", type="primary")
        if submitted:
            try:
                group, nearby = assign_product(bundle, title)
            except ValueError as error:
                st.warning(str(error))
            else:
                st.success(f"Assigned product group: {group}")
                st.write("Common words:", summary.set_index("Group").loc[group, "Common words"])
                st.subheader("Nearby offers in this group")
                st.dataframe(nearby, hide_index=True, width="stretch")
                st.caption("Groups describe text similarity. They are not verified product matches or predicted category labels.")
    else:
        catalog = bundle["hierarchical_catalog"]
        summary = bundle["hierarchical_summary"]
        silhouette = bundle["hierarchical_silhouette"]
        st.info(
            "Hierarchical clustering groups a fixed sample of 1,200 offers. It has no predict method for new titles; "
            "explore its existing groups below, or use K-Means for a new title."
        )
    columns = st.columns(3)
    columns[0].metric("Product offers grouped", f"{len(catalog):,}")
    columns[1].metric("Groups", "6")
    columns[2].metric("Sampled silhouette", f"{silhouette:.3f}")
    st.caption(
        "Both algorithms use product-title text only (TF-IDF and 32 reduced text components). "
        "IDs, known cluster labels, and categories are excluded from fitting. "
        "Silhouette is estimated on 1,000 offers; the two algorithms use different dataset sizes."
    )
    st.subheader("Explore a group")
    group = st.selectbox("Product group", summary["Group"].tolist(), key=f"group_{algorithm}")
    st.write("Common words:", summary.set_index("Group").loc[group, "Common words"])
    members = catalog[catalog["Group"].eq(group)]
    st.dataframe(members[["Product title", "Dataset category"]].head(20), hide_index=True, width="stretch")
    st.caption("Dataset category is shown for context only; it was not used to form the groups.")
    with st.expander("Group overview and map", expanded=True):
        st.dataframe(summary, hide_index=True, width="stretch")
        points = catalog.sample(n=min(1200, len(catalog)), random_state=42)
        st.scatter_chart(points, x="Component 1", y="Component 2", color="Group")
        st.caption("Two-dimensional projection of the text features. Group numbers are arbitrary identifiers.")


def render_classification():
    st.header("Identify a gas from a sensor reading")
    st.write("Select a saved reading to fill all 128 features automatically, or upload your own sensor CSV.")
    bundle = load_bundle("gas_classifier.joblib")
    if bundle.get("feature_columns") != SENSOR_COLUMNS:
        raise ValueError("The saved classifier feature order does not match Feature1 through Feature128.")
    mode = st.radio(
        "Sensor input", ["Sample sensor reading", "CSV upload", "Manual entry"],
        horizontal=True, key="sensor_mode",
    )
    values, actual_gas = None, None
    if mode == "Sample sensor reading":
        examples = load_sensor_examples()
        gas = st.selectbox("Example gas", examples["Gas"].unique().tolist(), key="example_gas")
        choices = examples[examples["Gas"].eq(gas)].reset_index(drop=True)
        index = st.selectbox(
            "Sensor reading", list(range(len(choices))),
            format_func=lambda i: f"Reading {i + 1} - {choices.iloc[i]['batch']}", key=f"reading_{gas}",
        )
        row = choices.iloc[index]
        values = row[SENSOR_COLUMNS].astype(float).tolist()
        actual_gas = row["Gas"]
        st.info("All 128 sensor values are loaded. Click Predict gas class to run the classifier.")
        st.caption("These are known dataset examples. The existing classifier was fitted on the full dataset, so this is a demonstration, not held-out evaluation.")
    elif mode == "CSV upload":
        st.download_button(
            "Download a sample sensor CSV", load_sensor_examples()[SENSOR_COLUMNS].head(1).to_csv(index=False),
            file_name="sensor_example.csv", mime="text/csv",
        )
        uploaded_file = st.file_uploader("CSV with Feature1 through Feature128", type=["csv"])
        if uploaded_file is not None:
            try:
                uploaded = pd.read_csv(uploaded_file)
                missing = [feature for feature in SENSOR_COLUMNS if feature not in uploaded.columns]
                if missing:
                    st.error(f"CSV is missing {len(missing)} required feature columns.")
                elif uploaded.empty:
                    st.error("The CSV does not contain any rows.")
                else:
                    values = pd.to_numeric(uploaded[SENSOR_COLUMNS].iloc[0], errors="raise").tolist()
                    st.caption("Using the first row of the uploaded CSV.")
            except (ValueError, UnicodeError, pd.errors.ParserError) as error:
                st.error(f"Could not read the sensor CSV: {error}")
    else:
        with st.expander("Enter 128 sensor features", expanded=True):
            columns = st.columns(4)
            values = []
            for index, feature in enumerate(SENSOR_COLUMNS):
                with columns[index % 4]:
                    values.append(st.number_input(feature, value=0.0, key=f"sensor_{feature}"))
    if values is not None:
        if not np.isfinite(np.asarray(values, dtype=float)).all():
            st.error("Sensor values must be finite numbers, without missing values or infinity.")
            return
        with st.expander("Inspect the 128 sensor values"):
            st.dataframe(pd.DataFrame({"Feature": SENSOR_COLUMNS, "Value": values}), hide_index=True, width="stretch")
        if st.button("Predict gas class", type="primary", key="predict_gas"):
            row = pd.DataFrame([values], columns=SENSOR_COLUMNS)
            model = bundle["model"]
            label = int(model.predict(row)[0])
            gas = bundle["gas_names"].get(label, str(label))
            st.success(f"Predicted gas: {gas}")
            if actual_gas is not None:
                st.write(f"Known example label: {actual_gas}")
            probabilities = pd.DataFrame({
                "Gas": [bundle["gas_names"].get(int(c), str(c)) for c in model.classes_],
                "Probability": model.predict_proba(row)[0],
            }).sort_values("Probability", ascending=False)
            st.bar_chart(probabilities.set_index("Gas"))
            st.dataframe(probabilities, hide_index=True, width="stretch")


def render_app():
    st.set_page_config(page_title="AutoML Project Lab", page_icon="🧪", layout="wide")
    st.title("AutoML Project Lab")
    st.caption("Three tasks. Real datasets. A demonstration you can run without typing 128 sensor values.")
    with st.sidebar:
        st.header("Choose a demonstration")
        task = st.radio("Task", ["Regression", "Clustering", "Classification"], key="task")
        st.divider()
        st.write("**Quick evaluator walkthrough**")
        st.write("1. Predict PM2.5 from a held-out weather example.\n2. Group a product title and inspect similar offers.\n3. Identify gas from a saved sensor reading.")
    try:
        {"Regression": render_regression, "Clustering": render_clustering,
         "Classification": render_classification}[task]()
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))
