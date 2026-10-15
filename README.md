# Gas Sensor Array Drift at Different Concentrations — Machine Learning Capstone

## 23CSE301 — Machine Learning Capstone Project

This project studies the application of Machine Learning techniques to **gas sensor array measurements** for gas identification and concentration prediction.

The dataset used in this project is the **Gas Sensor Array Drift at Different Concentrations Dataset** from the **UCI Machine Learning Repository**.

---

## 1. Dataset Description

The **Gas Sensor Array Drift at Different Concentrations** dataset contains measurements collected from **16 chemical sensors** exposed to **6 different gases** at various concentration levels.

According to the UCI Machine Learning Repository, the dataset contains:

* **13,910 measurements**
* **128 features**
* **16 chemical sensors**
* **6 gas classes**
* Real-valued sensor features
* No missing values
* Data collected over a period of **36 months**
* Data organized into **10 batches**

The six gases represented in the dataset are:

| Class | Gas          |
| ----: | ------------ |
|     1 | Ethanol      |
|     2 | Ethylene     |
|     3 | Ammonia      |
|     4 | Acetaldehyde |
|     5 | Acetone      |
|     6 | Toluene      |

The dataset additionally provides the **concentration level** associated with each measurement, making it suitable for both classification and regression tasks.

Source: UCI Machine Learning Repository
Dataset: **Gas Sensor Array Drift at Different Concentrations**
DOI: https://doi.org/10.24432/C5MK6M

---

## 2. Dataset Structure

Each observation corresponds to one measurement obtained from the sensor array.

The 128-dimensional feature vector is formed from **8 extracted features for each of the 16 sensors**.

The eight features associated with each sensor are:

1. `DR`
2. `|DR|`
3. `EMAi0.001`
4. `EMAi0.01`
5. `EMAi0.1`
6. `EMAd0.001`
7. `EMAd0.01`
8. `EMAd0.1`

Therefore:

```text
16 sensors × 8 features per sensor = 128 features
```

The features describe both steady-state sensor response and dynamic characteristics of the sensor response.

The original dataset is distributed as multiple batch files in `.dat` format.

---

## 3. Problem Statement

The project investigates how machine learning can be used to analyse the responses of a chemical sensor array.

The project is divided into three Machine Learning tracks:

### Regression

Predict the **gas concentration level** from the 128 sensor-derived features.

### Classification

Predict the **type of gas** from the sensor measurements.

### Clustering

Discover natural groups in the sensor measurements without using gas labels during model fitting.

The Regression and Classification tracks are evaluated in Review 1 and Review 2, while the complete Clustering track is evaluated in Review 2.

---

## 4. Review 1 Scope

The Review 1 implementation covers:

### Dataset and Exploratory Data Analysis

* Dataset loading
* Dataset shape and structure
* Data type inspection
* Missing-value analysis
* Duplicate analysis
* Outlier analysis
* Target distribution
* Feature distributions
* Correlation analysis
* Feature-target relationships

### Preprocessing and Feature Engineering

* Conversion of the original `.dat` format into a Pandas DataFrame
* Separation of input features and targets
* Duplicate checking
* Outlier analysis
* Feature scaling
* Train-test split
* Feature engineering
* Prevention of data leakage

### Regression

The following ten regression algorithms are implemented:

1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. ElasticNet Regression
5. Polynomial Regression
6. Decision Tree Regressor
7. Random Forest Regressor
8. Gradient Boosting Regressor
9. Support Vector Regressor
10. K-Nearest Neighbors Regressor

### Classification — Part A

The following five classification algorithms are implemented for Review 1:

1. Logistic Regression
2. K-Nearest Neighbors
3. Gaussian Naive Bayes
4. Decision Tree Classifier
5. Support Vector Classifier

---

## 5. Regression Objective

The regression task uses the **gas concentration level** as the target variable.

Conceptually:

```text
128 Sensor Features
        ↓
Preprocessing
        ↓
Feature Engineering
        ↓
Regression Model
        ↓
Predicted Gas Concentration
```

The regression models are evaluated using:

* R² Score
* Root Mean Squared Error (RMSE)
* Mean Absolute Error (MAE)

The two best-performing models are further evaluated using **5-fold cross-validation**.

---

## 6. Classification Objective

The classification task predicts the gas being measured.

```text
128 Sensor Features
        ↓
Preprocessing
        ↓
Classification Model
        ↓
Predicted Gas Type
```

The six classes are:

```text
1 → Ethanol
2 → Ethylene
3 → Ammonia
4 → Acetaldehyde
5 → Acetone
6 → Toluene
```

For Review 1, the following metrics are considered:

* Accuracy
* Weighted Precision
* Weighted Recall
* Weighted F1-score
* Confusion Matrix

The final Review 2 comparison now includes all ten classification algorithms, with weighted
Precision, weighted Recall, weighted F1, and multiclass ROC-AUC (OvR) reported in one table.

---

## 7. Exploratory Data Analysis

The EDA stage investigates:

### Dataset Structure

```python
df.shape
df.info()
df.describe()
```

### Missing Values

The UCI dataset reports **no missing values**.

Therefore, missing-value imputation is not required. Instead, the project verifies the absence of missing observations before continuing with preprocessing.

### Duplicate Records

Duplicate observations are checked and their presence or absence is reported.

### Outliers

Outliers are investigated using appropriate statistical and visual techniques such as:

* Box plots
* Interquartile Range (IQR)
* Distribution plots

Any treatment of outliers is documented and justified.

### Feature Distribution

Feature distributions are examined to understand:

* Range of sensor responses
* Skewness
* Variability
* Differences between sensor-derived features

### Correlation Analysis

A correlation matrix and heatmap are used to investigate relationships between numerical features.

### Target Distribution

The distributions of:

* Gas classes
* Concentration levels

are analysed to identify imbalance or uneven representation.

---

## 8. Preprocessing Pipeline

The general preprocessing pipeline is:

```text
Raw .dat files
      ↓
Parse observations
      ↓
Create DataFrame
      ↓
Separate features and targets
      ↓
Check missing values
      ↓
Check duplicates
      ↓
Check outliers
      ↓
Feature engineering
      ↓
Train-Test Split
      ↓
Fit scaler on training data
      ↓
Transform training and test data
      ↓
Train ML models
      ↓
Evaluate models
```

A fixed random state is used wherever applicable to ensure reproducibility.

All preprocessing transformations that learn parameters from the data are fitted **only on the training set** to avoid data leakage.

---

## 9. Feature Engineering

The original dataset contains 128 sensor-derived features.

Feature engineering will investigate the relationships between sensor measurements and may include sensor-wise aggregate or interaction features where appropriate.

Any engineered feature included in the final pipeline will be accompanied by an explanation of:

1. What the feature represents
2. Why it was created
3. Why it may improve model performance

Feature engineering decisions will be based on observations from EDA rather than arbitrary transformations.

---

## 10. Model Evaluation

All regression algorithms are evaluated using the **same preprocessed dataset and held-out test set** so that model comparisons remain fair.

### Regression Metrics

| Metric | Purpose                                    |
| ------ | ------------------------------------------ |
| R²     | Measures explained variance                |
| RMSE   | Penalizes large prediction errors          |
| MAE    | Measures average absolute prediction error |

The two best regression models are additionally evaluated using 5-fold cross-validation.

### Classification Metrics

| Metric           | Purpose                                   |
| ---------------- | ----------------------------------------- |
| Accuracy         | Overall proportion of correct predictions |
| Precision        | Correctness of positive predictions       |
| Recall           | Ability to identify samples of a class    |
| Weighted F1      | Balance between precision and recall      |
| Confusion Matrix | Class-wise prediction analysis            |

---

## 11. Required Visualisations

The project includes visualisations required by the capstone rubric, including:

* Feature distribution plots
* Target distribution
* Correlation heatmap
* Feature-target scatter plots where applicable
* Regression residual plot
* Predicted vs actual plot
* Tree-based feature importance
* Classification confusion matrix

All plots include suitable titles and labelled axes.

---

## 12. Technologies Used

The project is implemented using Python and the following libraries:

* Python 3
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

Additional packages may be included where required for dataset loading or preprocessing.

---

## 13. Project Structure

```text
gas-sensor-ml-capstone/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── batch1.dat
│   ├── batch2.dat
│   ├── ...
│   └── batch10.dat
│
├── notebooks/
│   ├── regression.ipynb
│   └── classification.ipynb
│
├── models/
│   └── saved_models/
│
└── app/
    └── ...
```

The Review 2 clustering notebook is available at `gas_sensor_ml_capstone/notebooks/clustering.ipynb`.
It uses `data/pricerunner_aggreagte.csv`, fits K-Means and Agglomerative Hierarchical Clustering,
reports the required validation metrics, and includes elbow, dendrogram, PCA, and t-SNE plots.

---

## 14. Dataset Source and Citation

### UCI Machine Learning Repository

Vergara, A. (2012). *Gas Sensor Array Drift at Different Concentrations*. UCI Machine Learning Repository.

DOI:

https://doi.org/10.24432/C5MK6M

Dataset page:

https://archive.ics.uci.edu/dataset/270/gas+sensor+array+drift+dataset+at+different+concentrations

The dataset is licensed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**.

The dataset documentation also states that the dataset is intended for research and requires citation of the associated works.

---

## 15. Reproducibility

To ensure reproducible results:

* A fixed random state is used wherever applicable.
* The same train-test split is used when comparing models within a track.
* Preprocessing is fitted only on training data.
* Model parameters and evaluation metrics are recorded.
* Required dependencies are listed in `requirements.txt`.

---

## 16. Academic Integrity

All analysis, interpretation, feature-engineering decisions, and conclusions are developed by the project team.

External sources and borrowed code are cited in the relevant notebook or README sections.

Generative AI assistance, where used, is limited according to the course guidelines and is disclosed in the repository.

---

## 17. Current Review 1 Status

### Completed / Planned for Review 1

* [x] Dataset selection
* [x] Dataset description
* [ ] Dataset loading and parsing
* [ ] Dataset audit
* [ ] Missing-value analysis
* [ ] Duplicate analysis
* [ ] Outlier analysis
* [ ] Exploratory Data Analysis
* [ ] Feature engineering
* [ ] Train-test split
* [ ] Feature scaling
* [ ] 10 regression algorithms
* [ ] Regression comparison table
* [ ] Hyperparameter tuning for at least 2 regression models
* [ ] Cross-validation for the best regression models
* [ ] Regression visualisations
* [ ] Classification Part A — 5 algorithms
* [ ] Classification evaluation

---

## 18. Project Goal

The overall goal of this project is to develop a reproducible machine learning pipeline capable of learning useful patterns from chemical gas sensor measurements while comparing multiple machine learning approaches across regression, classification, and clustering tasks.

The final project will compare model performance, analyse the effect of preprocessing and feature engineering, and identify the most suitable approaches for the selected prediction tasks.
