"""Train and persist the classification model used by the Streamlit app."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_loader import GAS_NAMES, load_dataset


RANDOM_STATE = 42
N_ESTIMATORS = 100


def train_and_save_model(project_root=None):
    """Train a reproducible gas-class classifier and save its deployment bundle."""
    project_root = Path(project_root or Path(__file__).resolve().parents[1])
    data_dir = project_root / "data"
    model_dir = project_root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    data = load_dataset(data_dir)
    feature_columns = [f"Feature{i}" for i in range(1, 129)]
    X = data[feature_columns].apply(pd.to_numeric, errors="coerce")
    y = data["gas_class"]

    if X.isna().any().any():
        X = X.fillna(X.median())

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=N_ESTIMATORS,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    pipeline.fit(X, y)

    bundle = {
        "model": pipeline,
        "feature_columns": feature_columns,
        "gas_names": GAS_NAMES,
        "model_name": "Random Forest Classifier",
        "random_state": RANDOM_STATE,
    }
    output_path = model_dir / "gas_classifier.joblib"
    joblib.dump(bundle, output_path, compress=3)
    print(f"Saved deployment model to {output_path}")
    print(f"Training rows: {len(data):,}")
    print(f"Classes: {sorted(y.unique().tolist())}")
    return output_path


if __name__ == "__main__":
    train_and_save_model()
