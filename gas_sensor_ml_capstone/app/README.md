# AutoML Project Lab

The app now has three demonstrations selected in the sidebar:

- **Regression:** predict Beijing PM2.5 with Random Forest or KNN. Use one of 12
  held-out examples, or edit six weather values. Temperature minus dew point is
  calculated automatically. Predictions on unchanged examples include the actual
  value and absolute error. Test R2, MAE, and RMSE are shown for each model.
- **Clustering:** type a product title to assign it to one of six K-Means groups
  and see nearby offers. Explore six hierarchical groups in a deterministic
  1,200-offer sample, with representative words and a group map. Hierarchical
  clustering does not predict groups for new titles.
- **Classification:** select a saved gas-sensor reading to populate all 128
  values automatically. CSV upload and manual entry remain available.

From the `gas_sensor_ml_capstone` directory:

```powershell
pip install -r requirements.txt
python src/train_app_model.py
python src/train_demo_models.py
streamlit run app/app.py
```

The model artifacts are already generated and should be committed along with the
app: `models/gas_classifier.joblib`, `models/regression_demo.joblib`,
`models/clustering_demo.joblib`, and `models/classification_examples.csv`.
Training runs offline; the deployed app loads the saved artifacts.

## Evaluation notes

Regression uses the notebook's six weather inputs plus `Temp_Dew_Diff`, an 80/20
random split with seed 42, and preprocessing fitted only on training rows. The
app models are newly trained demo versions, not exports of the notebook's tuned
estimators. The forest uses 100 trees, depth 18, and at least two samples per
leaf; KNN uses seven neighbors, distance weights, and Manhattan distance. A random
split is suitable for matching this project workflow, but is not a future-time
forecast evaluation. Models are serialized with scikit-learn 1.9.0, pinned in the
requirements file; regenerate artifacts if changing that dependency.

Clustering uses only product-title text (TF-IDF, 32 SVD components, normalization).
The notebook's numeric-ID fallback is not used: product IDs, merchant IDs, known
cluster IDs, categories, and labels are excluded from fitting. K-Means fits all
35,311 offers; Ward hierarchical clustering fits a fixed 1,200-offer sample to
bound memory. Silhouette estimates use 1,000 offers from each fitted dataset and
are not an apples-to-apples ranking of the algorithms. Dataset categories are
display metadata only. Group numbers do not represent known classes.

Classification examples come from the dataset used to fit the existing
classifier. They demonstrate the interface and are not independent test results.

Run the app and model checks from the repository root:

```powershell
python -m unittest discover -s gas_sensor_ml_capstone/tests -v
```

## Streamlit Community Cloud

Select your fork's `main` branch and use
`gas_sensor_ml_capstone/app/app.py` as the main file path.

The repository-root `requirements.txt` includes the existing
`gas_sensor_ml_capstone/requirements.txt`. Streamlit searches the entrypoint's
directory and the repository root, so the nested requirements file alone is not
discovered. Push the root file to the deployed fork and branch to trigger a
rebuild, then check the build logs for `joblib` and `scikit-learn`.

To test from the repository root:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run gas_sensor_ml_capstone/app/app.py
```

Use the Python and package versions used to train the saved model. Select Python
in Streamlit's Advanced settings; the empty `runtime.txt` does not set a version.

See [Streamlit's dependency documentation](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).
