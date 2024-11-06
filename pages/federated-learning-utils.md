---
title: Federated Learning Tutorial - Utilities
---

The `utils.py` file contains utility functions and classes to support the main federated learning process defined in the `server.py` file. Here is an explanation of its components:

### Imports
```python
from typing import Tuple, Union, List
import numpy as np
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
```
- Various libraries are imported, including `numpy` for numerical operations, `pandas` for data manipulation, and `sklearn` for machine learning utilities.

### Constants and Type Aliases
```python
DATA_DIR = "./data/"
DATA_DIR_SERVER = "./Data Center/"

XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]
```
- Constants for data directories.
- Type aliases for better code readability.

### Model Parameter Utilities
```python
def get_model_parameters(model):
    """Returns the parameters of a sklearn MLClassifier model"""
    params = (model.coefs_[0], model.coefs_[1], model.intercepts_[0], model.intercepts_[1])
    return params

def set_model_params(model: MLPClassifier, params):
    """Sets the parameters of a sklearn LogisticRegression model"""
    model.coefs_[0] = params[0]
    model.coefs_[1] = params[1]
    model.intercepts_[0] = params[2]
    model.intercepts_[1] = params[3]
    return model

def set_initial_params(model: MLPClassifier):
    """
    Sets initial parameters as zeros
    """
    # Placeholder function for setting initial parameters
```
- Functions to get and set model parameters, which are essential for federated learning where model parameters are exchanged.

### Feature Names Utility
```python
def get_var_names():
    return ["mean_texture", "mean_smoothness", "mean_area"]
```
- Returns a list of feature names used in the dataset.

### Data Loading Functions
```python
def load_test_data() -> Dataset:
    """
    Loads the test dataset from the central server.
    """
    df = pd.read_csv(DATA_DIR_SERVER + "central_test.csv")
    var_names = get_var_names()
    X_test = df[var_names].to_numpy()
    y_test = df[['target']].to_numpy()
    return X_test, y_test

def load_data() -> Dataset:
    """
    Loads and preprocesses the dataset for federated learning.
    """
    data = load_breast_cancer()
    df = pd.DataFrame(data["data"], columns=data["feature_names"])
    df["target"] = data["target"]
    df = df.iloc[:70]  # Limit the dataset size for some reason

    var_names = ['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area',
                 'mean_smoothness', 'mean_compactness', 'mean_concavity',
                 'mean_concave_points', 'mean_symmetry', 'mean_fractal_dimension', 'target']

    rename_dict = {'mean radius': "mean_radius",
                   'mean texture': "mean_texture",
                   'mean perimeter': "mean_perimeter",
                   'mean area': "mean_area",
                   'mean smoothness': "mean_smoothness",
                   'mean compactness': "mean_compactness",
                   'mean concavity': "mean_concavity",
                   'mean concave points': "mean_concave_points",
                   'mean symmetry': "mean_symmetry",
                   'mean fractal dimension': "mean_fractal_dimension",
                   'target': "target"}

    df = df.rename(columns=rename_dict)
    df.mean().to_csv(DATA_DIR + "mean.csv")
    df.std().to_csv(DATA_DIR + "std.csv")

    random_state = 42
    federated_1, remaining = train_test_split(df[var_names], test_size=0.8, random_state=random_state)
    federated_2, central = train_test_split(remaining, test_size=0.8, random_state=random_state)
    central_train, central_test = train_test_split(central, test_size=0.5, random_state=random_state)

    federated_central_1, federated_central_2 = train_test_split(central_train, test_size=0.5, random_state=random_state)

    federated_1.to_csv(DATA_DIR + "federated_1.csv", index=False)
    federated_2.to_csv(DATA_DIR + "federated_2.csv", index=False)
    central_train.to_csv(DATA_DIR + "federated_central.csv", index=False)
    central_test.to_csv(DATA_DIR + "central_test.csv", index=False)
    federated_central_1.to_csv(DATA_DIR + "federated_central_1.csv", index=False)
    federated_central_2.to_csv(DATA_DIR + "federated_central_2.csv", index=False)
```
- `load_test_data`: Loads test data from a CSV file located on the server.
- `load_data`: Loads and preprocesses the breast cancer dataset, renames columns, computes statistics, splits the data for federated learning, and saves the subsets as CSV files.

### Summary
- **Model Utilities**: Functions to get and set model parameters.
- **Data Utilities**: Functions to load and preprocess datasets, including splitting data for federated learning.
- **Constants and Type Aliases**: Used for code organization and readability.