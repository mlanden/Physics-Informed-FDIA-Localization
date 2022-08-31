import torch.nn as nn
from utils import activations


class PredictionModel(nn.Module):

    def __init__(self, conf):
        super(PredictionModel, self).__init__()

        hidden_layers = conf["model"]["hidden_layers"]
        n_features = conf["data"]["n_features"]

        self.activation = activations[conf["model"]["activation"]]
        self.linear = nn.Linear(n_features, hidden_layers[0])
        self.rnns = []
        for i in range(len(hidden_layers[:-1])):
            self.rnns.append(nn.LSTM(hidden_layers[i], hidden_layers[i + 1]))
        self.rnns.append(nn.LSTM(hidden_layers[-1], n_features))

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        for rnn in self.rnns:
            x, hidden = rnn(x)
            x = self.activation(x)
        return x
