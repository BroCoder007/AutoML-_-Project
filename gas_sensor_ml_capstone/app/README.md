# Gas Sensor Classifier App

This Streamlit app loads the persisted classification model from
`models/gas_classifier.joblib` and predicts a gas class from the 128 sensor features.

From the `gas_sensor_ml_capstone` directory:

```powershell
pip install -r requirements.txt
python src/train_app_model.py
streamlit run app/app.py
```

The app accepts a CSV upload containing `Feature1` through `Feature128`, or values entered
manually in the UI. The CSV option is recommended for testing real sensor measurements.

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
