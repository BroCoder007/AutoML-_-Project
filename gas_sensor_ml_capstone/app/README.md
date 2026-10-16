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
