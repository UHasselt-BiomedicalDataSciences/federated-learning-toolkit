---
title: Federated Learning Tutorial Hands on Experince
contributors: [Axel Faes, Ashkan Pirmani]
search_exclude: false
---

## Introduction

Welcome to this **hands-on tutorial** on Federated Learning (FL)! In this tutorial, you will gain practical experience in building and running a Federated Learning pipeline using the [Flower Framework](https://flower.dev/) and PyTorch. 

The tutorial consists of three main components:

1. **Client Script (`client.py`)**: Handles client-side logic, including local training and communication with the server.
2. **Server Script (`server.py`)**: Manages the central server for federated learning, including aggregation of client updates and saving the global model.
3. **Utilities Script (`utils.py`)**: Provides helper functions for data preprocessing, model utilities, and dataset management.

By the end of this tutorial, you will:
- Understand how federated learning operates in practice.
- Set up a federated learning pipeline.
- Train a simple neural network on distributed data.

### File Structure

The tutorial is structured into three key scripts:

1. **`client.py`**: Defines the client-side operations.
2. **`server.py`**: Manages the federated learning server.
3. **`utils.py`**: Contains utility functions for data and model handling.

---

## Client.py


The `client.py` file defines the client-side logic for a federated learning setup using the Flower (FL) framework. Here's a detailed breakdown of the code:

### Imports
```python
import flwr as fl
import utils
import argparse
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from collections import OrderedDict
```
- Imports necessary libraries, including Flower for federated learning, PyTorch for building and training neural networks, and utility functions from `utils.py`.

### Neural Network Definition
```python
class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.layer1 = nn.Linear(len(utils.get_var_names()), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        return torch.nn.functional.sigmoid(x)
```
- Defines a simple neural network with one linear layer followed by a sigmoid activation function. The number of input features is determined by the `get_var_names` function from `utils.py`.

### Training and Testing Functions
```python
def train(net, trainloader, epochs):
    """Train the network on the training set."""
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    net.train()
    for _ in range(epochs):
        for x, y in trainloader:
            optimizer.zero_grad()
            loss = criterion(net(x)[:,0], y)
            print(loss)
            loss.backward()
            optimizer.step()

def test(net, loader):
    criterion = torch.nn.CrossEntropyLoss()
    net.eval()
    loss = 0
    i = 1
    for x, y in loader:
        loss += criterion(net(x)[:,0], y)
        i += 1
    acc = 0
    return loss / i, acc
```
- `train`: Trains the neural network using binary cross-entropy loss and stochastic gradient descent (SGD) optimizer.
- `test`: Evaluates the neural network using cross-entropy loss (though it should be binary cross-entropy given the model's output and task).

### Data Loading Function
```python
def load_data(agent_id):
    """Load Data."""
    X, y = utils.load_data(agent_id)
    ds = TensorDataset(torch.Tensor(X), torch.Tensor(y))
    trainloader = DataLoader(ds, batch_size=16, shuffle=True)
    num_examples = len(y)
    return trainloader, num_examples
```
- Loads data using the `load_data` function from `utils.py`, converts it to PyTorch tensors, and returns a DataLoader for batching.

### Main Client Logic
```python
def main(agent_id, server_address):
    """Create model, load data, define Flower client, start Flower client."""
    # Load model
    net = Net()

    # Load data
    trainloader, num_examples = load_data(agent_id)

    # Flower client
    class TorchClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            """
            Extracts the parameters from the network
            """
            return [val.cpu().numpy() for _, val in net.state_dict().items()]

        def set_parameters(self, parameters):
            """
            Load params into the network
            """
            params_dict = zip(net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            train(net, trainloader, epochs=10)
            return self.get_parameters(config), num_examples, {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            loss, accuracy = test(net, trainloader)
            return float(loss), num_examples, {"accuracy": float(accuracy)}

    # Start client
    fl.client.start_client(
        server_address=server_address,
        client=TorchClient().to_client(),
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Client side.')
    parser.add_argument('--agent_id', type=str, help="ID of the client, only for testing purposes")
    parser.add_argument('--server_address', type=str, help="ID of the client, only for testing purposes", default="localhost:8889")

    args = parser.parse_args()
    print(args)
    main(**vars(args))
```
- Defines the main function to:
  - Create the neural network model.
  - Load data.
  - Define a Flower client (`TorchClient`) that handles parameter exchange, training, and evaluation.
  - Start the Flower client to connect to the federated learning server.

### Key Points:
1. **Neural Network**: A simple single-layer neural network defined using PyTorch.
2. **Training and Testing**: Functions to train and evaluate the model.
3. **Data Loading**: Loads data specific to each client (agent).
4. **Flower Client**: Defines a client for federated learning using Flower, implementing methods to get and set model parameters, train, and evaluate the model.
5. **Main Execution**: Parses command-line arguments, initializes the model and data, and starts the Flower client.

This client code allows for distributed model training across multiple clients in a federated learning setup. Each client trains locally on its data and communicates updates to a central server.


## Server.py

The code in the file `server.py` is a server-side implementation for federated learning using the Flower (FL) framework. Here's a breakdown of the code:

### Imports
```python
import flwr as fl
import utils
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from typing import Dict
import argparse
import numpy as np
import torch
from collections import OrderedDict
import pickle
import client

from flwr.server.server import Server, ServerConfig
from flwr.server.client_manager import SimpleClientManager
from flwr.common.parameter import parameters_to_ndarrays
```
- Various libraries are imported, including Flower for federated learning, sklearn for metrics, torch for neural networks, and others.

### Custom Federated Averaging Strategy
```python
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, num_rounds, **kwargs):
        super().__init__(**kwargs)
        self.num_rounds = num_rounds

    def aggregate_fit(self, rnd: int, results, failures):
        weights = super().aggregate_fit(rnd, results, failures)
        if rnd == self.num_rounds:
            if weights is not None:
                print(f"Saving weights...")
                with open('./weights.pkl', 'wb') as handle:
                    pickle.dump(weights, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return weights
```
- This class extends the standard Federated Averaging strategy to save the model weights after the final round of training.

### Utility Functions
```python
def fit_round(rnd: int) -> Dict:
    """Send round number to client"""
    return {"rnd": rnd}

def get_eval_fn(parameters):
    """Return an evaluation function for server-side evaluation."""
    (X_test, y_test) = utils.load_test_data()
    net = client.Net()
    criterion = torch.nn.BCELoss()

    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

    y_pred = net(torch.Tensor(X_test))
    loss = criterion(y_pred[:,0], torch.Tensor(y_test)[:,0])
    y_pred = y_pred.detach().numpy()
    accuracy = (y_test[:,0] == (y_pred > 0.5)).mean()
    return loss, {"accuracy": accuracy}
```
- `fit_round` function sends the round number to the client.
- `get_eval_fn` function defines the evaluation logic for the server, using test data to compute loss and accuracy.

### Main Execution
```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Server side.')
    parser.add_argument('--server_address', type=str, help="ID of the client, only for testing purposes", default="localhost:8889")

    args = parser.parse_args()
    num_rounds = 20

    strategy = SaveModelStrategy(
        min_available_clients=2,
        on_fit_config_fn=fit_round,
        num_rounds=num_rounds
    )

    client_manager = SimpleClientManager()
    server = Server(client_manager=client_manager, strategy=strategy)

    print("Starting server on ", args.server_address)
    fl.server.start_server(
        server_address=args.server_address,
        server=server,
    )

    with open('./weights.pkl', 'rb') as handle:
        weights = pickle.load(handle)

    parameters = parameters_to_weights(weights[0])
    net = client.Net()
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

    (X_test, y_test) = utils.load_test_data()
    y_pred = net(torch.Tensor(X_test))
    y_pred = y_pred.detach().numpy()
    accuracy = (y_test[:,0] == (y_pred > 0.5)).mean()

    auc = roc_auc_score(y_test[:,0], y_pred[:,0])
    print("Magic is done")
    print(f"Thank You For Using FL4E - Your Accuracy : {accuracy}")
    print("You can now download your FL4E-trained model and weights as static files.")
    print("The community will appreciate you if you can now share them back to the platform.")
    print("Hope To See You Soon")
    torch.save(net.state_dict(), "./model.pt")
```
- The main section initializes the server, parses command-line arguments, sets up the federated learning strategy, and starts the server.
- After training, it loads the saved weights, updates the model, and evaluates it on the test data, printing the final accuracy and AUC score.

### Key Points:
1. **Custom Strategy**: `SaveModelStrategy` saves model weights after the final training round.
2. **Server Initialization**: Sets up a federated learning server with specified strategy and client manager.
3. **Evaluation**: Post-hoc evaluation using the saved model weights to determine accuracy and AUC on test data.
4. **Model Saving**: Final trained model is saved as `model.pt` for later use or sharing.


## Utils.py

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