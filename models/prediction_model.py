from typing import Tuple, Union

import torch
import torch.nn as nn
from utils import activations


class PredictionModel(nn.Module):

    def __init__(self, conf):
        super(PredictionModel, self).__init__()

        hidden_layers = conf["model"]["hidden_layers"]
        n_features = conf["data"]["n_features"]

        self.activation = activations[conf["model"]["activation"]]
        self.input_linear = nn.Linear(n_features, hidden_layers[0])
        self.rnns = []
        for i in range(len(hidden_layers[:-1])):
            self.rnns.append(nn.LSTM(hidden_layers[i], hidden_layers[i + 1], batch_first = True))
        self.rnns = nn.ModuleList(self.rnns)
        self.output_linear = nn.Linear(hidden_layers[-1], n_features)

    def forward(self, x, hidden_states = None) -> Union[Tuple[torch.Tensor, Tuple], torch.Tensor]:
        x = self.input_linear(x)
        x = self.activation(x)
        for i, rnn in enumerate(self.rnns):
            hidden = hidden_states[i] if hidden_states is not None else None
            x, hidden = rnn(x, hidden)
            if hidden_states is not None:
                hidden_states[i] = hidden
            x = self.activation(x)
        x = self.output_linear(x)
        if hidden_states is not None:
            return x, hidden_states
        else:
            return x
