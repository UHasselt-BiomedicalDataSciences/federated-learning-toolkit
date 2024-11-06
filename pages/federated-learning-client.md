---
title: Federated Learning Tutorial - Client
---

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