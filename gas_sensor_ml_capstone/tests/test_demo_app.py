"""Checks for evaluation integrity and the app's main demonstration paths."""

import sys
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from demo_models import WEATHER_COLUMNS, assign_product, weather_features


class DemoAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.regression = joblib.load(PROJECT_ROOT / "models/regression_demo.joblib")
        cls.clustering = joblib.load(PROJECT_ROOT / "models/clustering_demo.joblib")

    def start_app(self):
        app = AppTest.from_file(str(PROJECT_ROOT / "app/app.py")).run(timeout=45)
        self.assert_clean(app)
        return app

    def assert_clean(self, app):
        self.assertFalse(app.exception, [item.message for item in app.exception])
        self.assertFalse(app.error, [item.value for item in app.error])

    def test_regression_examples_are_held_out_and_metrics_reproduce(self):
        bundle = self.regression
        training = set(bundle["train_indices"])
        testing = set(bundle["test_indices"])
        self.assertFalse(training & testing)
        self.assertTrue(set(bundle["examples"]["Source row"]) <= testing)
        data = pd.read_csv(PROJECT_ROOT / "data/beijing+pm2+5+data/PRSA_data_2010.1.1-2014.12.31.csv")
        test_rows = data.loc[bundle["test_indices"]]
        for name, model in bundle["models"].items():
            predicted = model.predict(weather_features(test_rows))
            scores = bundle["metrics"].set_index("Model").loc[name]
            self.assertAlmostEqual(scores["R2"], r2_score(test_rows["pm2.5"], predicted))
            self.assertAlmostEqual(scores["MAE"], mean_absolute_error(test_rows["pm2.5"], predicted))

    def test_both_regression_models_predict_held_out_example(self):
        app = self.start_app()
        example = self.regression["examples"].iloc[0]
        for name, model in self.regression["models"].items():
            app.selectbox(key="regression_model").set_value(name).run(timeout=45)
            app.button(key="FormSubmitter:weather_prediction-Predict PM2.5").click().run(timeout=45)
            self.assert_clean(app)
            expected = float(model.predict(weather_features(pd.DataFrame([example])))[0])
            self.assertIn(f"{expected:.1f}", app.success[0].value)
            self.assertIn("Recorded PM2.5", [item.label for item in app.metric])

    def test_editing_weather_requires_six_inputs_and_no_known_target(self):
        app = self.start_app()
        app.radio(key="weather_mode").set_value("Edit weather values").run(timeout=45)
        self.assertEqual(len(app.number_input), 6)
        app.number_input(key="weather_TEMP_0").set_value(20.0)
        app.button(key="FormSubmitter:weather_prediction-Predict PM2.5").click().run(timeout=45)
        self.assert_clean(app)
        self.assertTrue(app.success)
        self.assertNotIn("Recorded PM2.5", [item.label for item in app.metric])

    def test_product_assignment_and_hierarchical_explorer(self):
        app = self.start_app()
        app.radio(key="task").set_value("Clustering").run(timeout=45)
        app.button(key="FormSubmitter:product_assignment-Find product group").click().run(timeout=45)
        self.assert_clean(app)
        self.assertIn("Assigned product group", app.success[0].value)
        app.radio(key="clustering_algorithm").set_value("Agglomerative hierarchical").run(timeout=45)
        self.assert_clean(app)
        self.assertTrue(any("no predict method" in item.value for item in app.info))
        self.assertFalse(app.text_input)
        self.assertEqual(app.metric[0].value, "1,200")
        app.selectbox(key="group_Agglomerative hierarchical").select_index(1).run(timeout=45)
        self.assert_clean(app)

    def test_unknown_product_text_has_helpful_message(self):
        app = self.start_app()
        app.radio(key="task").set_value("Clustering").run(timeout=45)
        app.text_input(key="product_title").set_value("zzzzzzqqqqvvvv")
        app.button(key="FormSubmitter:product_assignment-Find product group").click().run(timeout=45)
        self.assert_clean(app)
        self.assertTrue(app.warning)
        self.assertFalse(app.success)

    def test_saved_sensor_reading_needs_no_manual_values(self):
        app = self.start_app()
        app.radio(key="task").set_value("Classification").run(timeout=45)
        self.assertEqual(len(app.number_input), 0)
        self.assertTrue(any("not held-out evaluation" in item.value for item in app.caption))
        app.button(key="predict_gas").click().run(timeout=45)
        self.assert_clean(app)
        self.assertIn("Predicted gas:", app.success[0].value)
        app.selectbox(key="example_gas").select_index(1).run(timeout=45)
        app.button(key="predict_gas").click().run(timeout=45)
        self.assert_clean(app)
        self.assertTrue(app.success)

    def test_invalid_inputs_and_title_only_clustering(self):
        with self.assertRaisesRegex(ValueError, "familiar product words"):
            assign_product(self.clustering, "zzzzzzqqqqvvvv")
        with self.assertRaisesRegex(ValueError, "title first"):
            assign_product(self.clustering, "   ")
        row = {feature: 0.0 for feature in WEATHER_COLUMNS}
        row["TEMP"] = np.inf
        with self.assertRaisesRegex(ValueError, "finite"):
            weather_features(pd.DataFrame([row]))
        vocabulary = self.clustering["vectorizer"].get_feature_names_out()
        self.assertNotIn("category label", vocabulary)
        group, nearby = assign_product(self.clustering, "apple iphone 8 silver")
        self.assertIn(group, self.clustering["catalog"]["Group"].unique())
        self.assertEqual(len(nearby), 5)


if __name__ == "__main__":
    unittest.main()
