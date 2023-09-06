
import torch
import torch.nn as nn

from .ics_model import ICSModel


class PredictionModel(ICSModel):

    def __init__(self, conf, categorical_values):
        super().__init__(conf, categorical_values)

        self.n_layers = conf["model"]["n_layers"]
        self.hidden_size = conf["model"]["hidden_size"]

        self.create_layers(self.hidden_size)
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, self.n_layers, batch_first=True, dropout=self.dropout)

    def forward(self, unscaled_seq, scaled_seq):
        x = self.embed(unscaled_seq, scaled_seq)
        x = self.input_linear(x)
        x = self.activation(x)

        hx = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(torch.float32).to(x.device)
        cx = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(torch.float32).to(x.device)
        out, _ = self.rnn(x)
        out = out[:, -1, :]

        return self.outputs(out)

