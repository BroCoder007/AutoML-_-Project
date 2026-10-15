# Gas Sensor Array Drift at Different Concentrations — ML Capstone

## Project Structure

- `data/` — dataset files
- `notebooks/regression.ipynb` — Review 1 regression work
- `notebooks/classification.ipynb` — Review 1 Classification Part A + Review 2 Part B
- `notebooks/clustering.ipynb` — Review 2 PriceRunner clustering track
- `src/data_loader.py` — dataset loading/parsing utilities
- `requirements.txt` — Python dependencies

## Review 2

The classification notebook contains all ten required algorithms, a consolidated comparison
table with Accuracy, weighted Precision, weighted Recall, weighted F1, and multiclass ROC-AUC,
hyperparameter tuning, cross-validation, and final model selection.

The clustering notebook uses `data/pricerunner_aggreagte.csv` and implements both required
algorithms:

- K-Means, including an elbow curve
- Agglomerative Hierarchical Clustering, including a dendrogram

It also reports Silhouette, Davies-Bouldin, and Calinski-Harabasz metrics and includes PCA
cluster plots plus a t-SNE visualization. Ground-truth labels, if present in the CSV, are not
used to fit the clustering models.

To run the notebooks:

```powershell
pip install -r requirements.txt
jupyter notebook notebooks
```

Place `pricerunner_aggreagte.csv` in `data/` before running `clustering.ipynb`.

## Bonus Streamlit App

The optional gas-classification demo is in `app/app.py`. It loads the persisted model from
`models/gas_classifier.joblib` and accepts either a CSV containing `Feature1` through
`Feature128` or manual feature values.

Run it locally from this directory:

```powershell
python src/train_app_model.py
streamlit run app/app.py
```

For Streamlit Community Cloud, use `gas_sensor_ml_capstone/app/app.py` as the main file path.

## Team Responsibility

- Regression: Team Member 1
- Classification Part A + common EDA/preprocessing: Team Member 2
- Dataset understanding, feature-engineering decisions, documentation and GitHub: Shared

## Dataset

UCI Machine Learning Repository — Gas Sensor Array Drift at Different Concentrations:
https://archive.ics.uci.edu/dataset/270/gas+sensor+array+drift+dataset+at+different+concentrations

Add the downloaded dataset files to `data/`.
