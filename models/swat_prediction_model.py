import joblib
from typing import Tuple, Union
from os import path

import torch
import torch.nn as nn
from utils import activations

from .prediction_model import PredictionModel

# idx of value: number of possibilities
CATEGORICAL_VALUES = {2: 3, 3: 2, 4: 2, 9: 3, 10: 2, 11: 2, 12: 2, 13: 2, 14: 2, 15: 2, 19: 3, 20: 3, 21: 3, 22: 3,
                      23: 2, 24: 2, 29: 2, 30: 2, 31: 2, 32: 2, 33: 3, 42: 2, 43: 2, 48: 2, 49: 2, 50: 2}


class SwatPredictionModel(PredictionModel):

    def __init__(self, conf):
        super(SwatPredictionModel, self).__init__()
        self.checkpoint = conf["train"]["checkpoint"]
        self.scalar_path = path.join("checkpoint", self.checkpoint, "scaler.gz")
        self.scalar = None

        hidden_layers = conf["model"]["hidden_layers"]
        n_features = conf["data"]["n_features"]
        self.embedding_size = conf["model"]["embedding_size"]
        self.mse_loss_fn = nn.MSELoss()

        self.activation = activations[conf["model"]["activation"]]
        self.input_linear = nn.Linear(n_features + len(CATEGORICAL_VALUES) * (self.embedding_size - 1),
                                      hidden_layers[0])
        self.embeddings = nn.ModuleList()
        for size in CATEGORICAL_VALUES.values():
            self.embeddings.append(nn.Embedding(size, self.embedding_size))
        self.rnns = nn.ModuleList()
        for i in range(len(hidden_layers[:-1])):
            self.rnns.append(nn.LSTM(hidden_layers[i], hidden_layers[i + 1], batch_first = True))

        self.output_linear = nn.Linear(hidden_layers[-1], n_features + len(CATEGORICAL_VALUES) *
                                       (self.embedding_size - 1), hidden_layers[0])

    def forward(self, x, hidden_states = None) -> Union[Tuple[torch.Tensor, Tuple], torch.Tensor]:
        x = self.embed(x)
        x = self.input_linear(x)
        x = self.activation(x)
        for i, rnn in enumerate(self.rnns):
            hidden = hidden_states[i] if hidden_states is not None else None
            x, hidden = rnn(x, hidden)
            if hidden_states is not None:
                hidden_states[i] = hidden
            x = self.activation(x)
        x = self.output_linear(x)
        x = self.reverse_embed(x)
        if hidden_states is not None:
            return x, hidden_states
        else:
            return x

    def loss(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse_loss = self.mse_loss_fn(predicted, target)
        return mse_loss

    def embed(self, batch: torch.Tensor) -> torch.Tensor:
        if self.scalar is None:
            self.scalar = joblib.load(self.scalar_path)

        new_batch = torch.zeros((batch.shape[0], batch.shape[1], batch.shape[2]
                                 + len(CATEGORICAL_VALUES) * (self.embedding_size - 1)))
        for i in range(len(batch)):
            scaled = self.scalar.transform(batch[i, ...])
            for j in range(batch.shape[1]):
                new_idx = 0
                embed_idx = 0
                for k in range(batch.shape[2]):
                    if k in CATEGORICAL_VALUES:
                        in_ = batch[i, j, k].detach().int()
                        if CATEGORICAL_VALUES[k] == 2:
                            in_ -= 1
                        new_batch[i, j, new_idx: new_idx + self.embedding_size] = self.embeddings[embed_idx](in_)
                        embed_idx += 1
                        new_idx += self.embedding_size
                    else:
                        new_batch[i, j, new_idx] = scaled[j, k]
                        new_idx += 1
        return new_batch

    def reverse_embed(self, batch: torch.Tensor) -> torch.Tensor:
        new_batch = torch.zeros((batch.shape[0], batch.shape[1], batch.shape[2]
                                 - len(CATEGORICAL_VALUES) * (self.embedding_size - 1)))
        for i in range(len(batch)):
            # Inverse transform?
            for j in range(batch.shape[1]):
                k = 0
                embedding_idx = 0
                new_idx = 0
                while k < batch.shape[2]:
                    if new_idx in CATEGORICAL_VALUES:
                        sample = batch[i, j, k: k + self.embedding_size].clone().detach().requires_grad_(True)
                        distance = torch.norm(self.embeddings[embedding_idx].weight.data - sample, dim = 1)
                        category = torch.argmin(distance)
                        if CATEGORICAL_VALUES[new_idx] == 2:
                            category += 1
                        new_batch[i, j, new_idx] = category
                        k += self.embedding_size
                        embedding_idx += 1
                    else:
                        new_batch[i, j, new_idx] = batch[i, j, k].clone().detach().requires_grad_(True)
                        k += 1
                    new_idx += 1
        return new_batch