import torch
import torch.nn as nn

from utils import activations


class FCN(nn.Module):

    def __init__(self, conf, n_inputs, n_outputs) -> None:
        super().__init__()

        self.n_layers = conf["model"]["n_layers"]
        self.hidden_size = conf["model"]["hidden_size"]

        self.activation = activations[conf["model"]["activation"]]
        self.layers = []
        for i in range(self.n_layers - 1):
            self.layers.append(nn.Linear(n_inputs if i == 0 else self.hidden_size, self.hidden_size))
            self.layers.append(self.activation)
        self.layers.append(nn.Linear(self.hidden_size, n_outputs))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)