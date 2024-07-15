# structure

drift_control/
├── drift_control/
│   ├── __init__.py
│   ├── drift_detector.py
│   ├── alert.py
│   ├── utils.py
├── tests/
│   ├── __init__.py
│   ├── test_drift_detector.py
│   ├── test_alert.py
│   ├── test_utils.py
├── setup.py
├── README.md
├── LICENSE
└── requirements.txt

# Drift Control

A package for monitoring and controlling data drift in machine learning models.

## Installation

```bash
pip install drift_control
```

## Example Usage

```python
import pandas as pd
from DataDriftDetector import DataDriftDetector

# Load your datasets
df_prior = pd.read_csv('path/to/prior_dataset.csv')
df_post = pd.read_csv('path/to/post_dataset.csv')

# Initialize the detector
drift_detector = DataDriftDetector(df_prior, df_post)

# Calculate data drift
drift_results = drift_detector.calculate_drift()
print(drift_results)

# Plot categorical to numeric relationships
categorical_to_numeric_plot = drift_detector.plot_categorical_to_numeric()
categorical_to_numeric_plot.savefig('categorical_to_numeric_plot.png')

# Plot numeric to numeric relationships
numeric_to_numeric_plot = drift_detector.plot_numeric_to_numeric()
numeric_to_numeric_plot.savefig('numeric_to_numeric_plot.png')

# Plot categorical distributions
categorical_plot = drift_detector.plot_categorical()
categorical_plot.savefig('categorical_plot.png')

# Compare ML efficacy
target_column = 'target'
ml_report = drift_detector.compare_ml_efficacy(target_column)
print(ml_report)
```

## Monitoring Data Drift in a Customer Churn Prediction Model

Assume you have two datasets:

churn_data_prior.csv - Historical data used to train a customer churn prediction model.
churn_data_post.csv - Recent data collected from current customers.
You want to monitor data drift and ensure your model's efficacy remains high.

```python
import pandas as pd
from DataDriftDetector import DataDriftDetector

# Load your datasets
df_prior = pd.read_csv('path/to/churn_data_prior.csv')
df_post = pd.read_csv('path/to/churn_data_post.csv')

# Initialize the detector
drift_detector = DataDriftDetector(df_prior, df_post)

# Calculate data drift
drift_results = drift_detector.calculate_drift()
print("Drift Results:")
print(drift_results)

# Plot categorical to numeric relationships
categorical_to_numeric_plot = drift_detector.plot_categorical_to_numeric()
categorical_to_numeric_plot.savefig('churn_categorical_to_numeric_plot.png')

# Plot numeric to numeric relationships
numeric_to_numeric_plot = drift_detector.plot_numeric_to_numeric()
numeric_to_numeric_plot.savefig('churn_numeric_to_numeric_plot.png')

# Plot categorical distributions
categorical_plot = drift_detector.plot_categorical()
categorical_plot.savefig('churn_categorical_plot.png')

# Compare ML efficacy
target_column = 'churn'
ml_report = drift_detector.compare_ml_efficacy(target_column)
print("ML Efficacy Report:")
print(ml_report)
```
