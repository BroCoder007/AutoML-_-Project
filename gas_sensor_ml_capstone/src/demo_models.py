"""Train and use the small, persisted models behind the evaluator demos."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler


RANDOM_STATE = 42
WEATHER_COLUMNS = ["DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]
REGRESSION_FEATURES = WEATHER_COLUMNS + ["Temp_Dew_Diff"]
SENSOR_COLUMNS = [f"Feature{i}" for i in range(1, 129)]


def weather_features(rows):
    """Keep the notebook's inputs and derive temperature minus dew point."""
    features = rows[WEATHER_COLUMNS].apply(pd.to_numeric, errors="raise").copy()
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Weather inputs must be finite numbers.")
    features["Temp_Dew_Diff"] = features["TEMP"] - features["DEWP"]
    return features[REGRESSION_FEATURES]


def build_regression_demo(project_root):
    path = Path(project_root) / "data/beijing+pm2+5+data/PRSA_data_2010.1.1-2014.12.31.csv"
    data = pd.read_csv(path).dropna(subset=["pm2.5"])
    X = weather_features(data)
    y = data["pm2.5"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    estimators = {
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=18, min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=2,
        ),
        "K-Nearest Neighbors": KNeighborsRegressor(
            n_neighbors=7, weights="distance", p=1, n_jobs=2,
        ),
    }
    models, scores = {}, []
    for name, estimator in estimators.items():
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", estimator),
        ])
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        models[name] = model
        scores.append({
            "Model": name,
            "R2": float(r2_score(y_test, predicted)),
            "MAE": float(mean_absolute_error(y_test, predicted)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, predicted))),
        })
    # These examples are strictly from the held-out partition.
    example_indices = X_test.sample(n=12, random_state=RANDOM_STATE).index
    examples = data.loc[example_indices, WEATHER_COLUMNS].copy()
    examples["Actual PM2.5"] = y_test.loc[example_indices]
    examples["Observed at"] = pd.to_datetime(
        data.loc[example_indices, ["year", "month", "day", "hour"]]
    ).dt.strftime("%Y-%m-%d %H:%M")
    chart_rows = X_test.sample(n=200, random_state=RANDOM_STATE)
    comparison = pd.DataFrame({"Actual PM2.5": y_test.loc[chart_rows.index]})
    for name, model in models.items():
        comparison[name] = model.predict(chart_rows)
    return {
        "models": models, "metrics": pd.DataFrame(scores),
        "examples": examples.reset_index(names="Source row"),
        "comparison": comparison.reset_index(drop=True),
        "training_rows": len(X_train), "test_rows": len(X_test),
        "train_indices": X_train.index.to_numpy(),
        "test_indices": X_test.index.to_numpy(),
        "sklearn_version": sklearn.__version__,
    }


def group_summary(titles, matrix, labels, terms):
    groups = []
    for group in sorted(np.unique(labels)):
        mask = labels == group
        weights = np.asarray(matrix[mask].sum(axis=0)).ravel()
        top_terms = terms[np.argsort(weights)[-6:][::-1]]
        groups.append({
            "Group": f"Group {int(group) + 1}", "Offers": int(mask.sum()),
            "Common words": ", ".join(top_terms),
            "Example title": titles.loc[mask].iloc[0],
        })
    return pd.DataFrame(groups)


def build_clustering_demo(project_root):
    path = Path(project_root) / "data/pricerunner_aggregate.csv"
    data = pd.read_csv(path)
    data.columns = data.columns.str.strip()
    data = data.dropna(subset=["Product Title"])
    data = data[data["Product Title"].str.strip().ne("")].reset_index(drop=True)
    titles = data["Product Title"].astype(str)
    # Never fit on Product ID, Merchant ID, Cluster ID, Category ID, or labels.
    vectorizer = TfidfVectorizer(
        strip_accents="unicode", lowercase=True, stop_words="english",
        ngram_range=(1, 2), min_df=2, max_features=5000, sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(titles)
    reducer = Pipeline([
        ("svd", TruncatedSVD(n_components=32, random_state=RANDOM_STATE)),
        ("normalizer", Normalizer()),
    ])
    embedding = reducer.fit_transform(matrix)
    kmeans = KMeans(n_clusters=6, n_init=10, random_state=RANDOM_STATE)
    labels = kmeans.fit_predict(embedding)
    projection = PCA(n_components=2, random_state=RANDOM_STATE).fit(embedding)
    coordinates = projection.transform(embedding)
    catalog = pd.DataFrame({
        "Product title": titles, "Dataset category": data["Category Label"],
        "Group": [f"Group {label + 1}" for label in labels],
        "Component 1": coordinates[:, 0], "Component 2": coordinates[:, 1],
    })
    rng = np.random.default_rng(RANDOM_STATE)
    sample_indices = np.sort(rng.choice(len(data), size=min(1200, len(data)), replace=False))
    hierarchical = AgglomerativeClustering(n_clusters=6, linkage="ward")
    hierarchical_labels = hierarchical.fit_predict(embedding[sample_indices])
    hierarchical_catalog = catalog.iloc[sample_indices].reset_index(drop=True).copy()
    hierarchical_catalog["Group"] = [f"Group {label + 1}" for label in hierarchical_labels]
    terms = vectorizer.get_feature_names_out()
    return {
        "vectorizer": vectorizer, "reducer": reducer, "kmeans": kmeans,
        "projection": projection, "embedding": embedding, "catalog": catalog,
        "hierarchical_catalog": hierarchical_catalog,
        "kmeans_summary": group_summary(titles, matrix, labels, terms),
        "hierarchical_summary": group_summary(
            titles.iloc[sample_indices].reset_index(drop=True),
            matrix[sample_indices], hierarchical_labels, terms,
        ),
        "kmeans_silhouette": float(silhouette_score(
            embedding, labels, sample_size=1000, random_state=RANDOM_STATE,
        )),
        "hierarchical_silhouette": float(silhouette_score(
            embedding[sample_indices], hierarchical_labels,
            sample_size=1000, random_state=RANDOM_STATE,
        )),
        "sklearn_version": sklearn.__version__,
    }


def assign_product(bundle, title):
    if not isinstance(title, str) or not title.strip():
        raise ValueError("Enter a product title first.")
    vector = bundle["vectorizer"].transform([title])
    if vector.nnz == 0:
        raise ValueError("No familiar product words found. Try a title such as 'apple iphone 8 silver'.")
    embedded = bundle["reducer"].transform(vector)
    label = int(bundle["kmeans"].predict(embedded)[0])
    group = f"Group {label + 1}"
    indices = np.flatnonzero(bundle["catalog"]["Group"].eq(group).to_numpy())
    distances = np.linalg.norm(bundle["embedding"][indices] - embedded[0], axis=1)
    nearest = indices[np.argsort(distances)[:5]]
    return group, bundle["catalog"].iloc[nearest][["Product title", "Dataset category"]]


def classification_examples(project_root):
    from data_loader import GAS_NAMES, load_dataset

    data = load_dataset(Path(project_root) / "data")
    samples = pd.concat([
        group.sample(n=min(3, len(group)), random_state=RANDOM_STATE)
        for _, group in data.groupby("gas_class")
    ]).reset_index(drop=True)
    samples["Gas"] = samples["gas_class"].map(GAS_NAMES)
    return samples[["Gas", "gas_class", "batch"] + SENSOR_COLUMNS]


def save_demo_models(project_root):
    project_root = Path(project_root)
    model_dir = project_root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for name, builder in [
        ("regression_demo.joblib", build_regression_demo),
        ("clustering_demo.joblib", build_clustering_demo),
    ]:
        bundle = builder(project_root)
        joblib.dump(bundle, model_dir / name, compress=3)
        print(f"Saved {name}", flush=True)
        if "metrics" in bundle:
            print(bundle["metrics"].to_string(index=False), flush=True)
    classification_examples(project_root).to_csv(model_dir / "classification_examples.csv", index=False)
    print("Saved classification_examples.csv", flush=True)
