---
title: Federated Learning Tutorial - Server
---

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