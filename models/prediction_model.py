import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from typing import Tuple, Union
from os import path

from utils import activations


class PredictionModel(nn.Module):

    def __init__(self, conf, categorical_values):
        super().__init__()
        self.checkpoint = conf["train"]["checkpoint"]
        self.scalar_path = path.join("checkpoint", self.checkpoint, "scaler.gz")
        self.scalar = None
        self.categorical_values = categorical_values

        hidden_layers = conf["model"]["hidden_layers"]
        self.n_features = conf["data"]["n_features"]
        self.embedding_size = conf["model"]["embedding_size"]

        self.activation = activations[conf["model"]["activation"]]
        if self.embedding_size > 0:
            input_len = self.n_features + len(self.categorical_values) * (self.embedding_size - 1)
        else:
            input_len = self.n_features + sum(self.categorical_values.values()) - len(self.categorical_values)
            self.input_linear = nn.Linear(input_len, hidden_layers[0])
        self.embeddings = nn.ModuleList()
        self.classifications = nn.ModuleList()
        for size in self.categorical_values.values():
            if self.embedding_size > 0:
                self.embeddings.append(nn.Embedding(size, self.embedding_size))
            self.classifications.append(nn.Linear(hidden_layers[-1], size))

        self.rnns = nn.ModuleList()
        for i in range(len(hidden_layers[:-1])):
            self.rnns.append(nn.LSTM(hidden_layers[i], hidden_layers[i + 1], batch_first=True))

        self.output_linear = nn.Linear(hidden_layers[-1], self.n_features - len(self.categorical_values))

    def forward(self, x, hidden_states=None):
        x = self.embed(x)
        x = self.input_linear(x)
        x = self.activation(x)
        for i, rnn in enumerate(self.rnns):
            hidden = hidden_states[i] if hidden_states is not None else None
            x, hidden = rnn(x, hidden)
            if hidden_states is not None:
                hidden_states[i] = hidden
            x = self.activation(x)

        x = x[:, -1, :]
        continuous_outputs = self.output_linear(x)
        outputs = [continuous_outputs]
        for layer in self.classifications:
            out = layer(x)
            outputs.append(out)

        return outputs, hidden_states

    def scale(self, target):
        if self.scalar is None:
            self.scalar = joblib.load(self.scalar_path)
        scaled_target = torch.tensor(self.scalar.transform(target), dtype=torch.float)
        return scaled_target

    def predict(self, batch: torch.Tensor, hidden_states: torch.Tensor = None) -> Union[
        Tuple[torch.Tensor, Tuple], torch.Tensor]:
        intermediate, hidden_states = self.forward(batch, hidden_states)

        output = torch.zeros(batch.shape[1:])
        continuous_outputs = self.output_linear(intermediate)
        continuous_outputs = self._inverse_scale(continuous_outputs)
        continuous_idx = 0
        classification_idx = 0
        for i in range(batch.shape[-1]):
            if i in self.categorical_values:
                logits = self.classifications[classification_idx](intermediate)
                class_ = torch.argmax(logits, dim=1)
                output[:, i] = class_
                classification_idx += 1
            else:
                # Previous state plus the predicted difference
                output[:, i] = batch[:, -1, continuous_idx] + continuous_outputs[:, continuous_idx]
                continuous_idx += 1

        if hidden_states is not None:
            return output, hidden_states
        else:
            return output

    def embed(self, batch: torch.Tensor) -> torch.Tensor:
        if self.scalar is None:
            self.scalar = joblib.load(self.scalar_path)

        new_batch = torch.zeros((batch.shape[0], batch.shape[1], self.input_linear.in_features))
        for i in range(len(batch)):
            scaled = self.scalar.transform(batch[i, ...])
            for j in range(batch.shape[1]):
                new_idx = 0
                embed_idx = 0
                for k in range(batch.shape[2]):
                    if k in self.categorical_values:
                        in_ = batch[i, j, k].detach().long()
                        if self.categorical_values[k] == 2:
                            in_ -= 1
                        if self.embedding_size > 0:
                            new_data = self.embeddings[embed_idx](in_)
                            size = self.embedding_size
                        else:
                            new_data = F.one_hot(in_, num_classes=self.categorical_values[k])
                            size = self.categorical_values[k]
                        new_batch[i, j, new_idx: new_idx + size] = new_data
                        embed_idx += 1
                        new_idx += size
                    else:
                        new_batch[i, j, new_idx] = scaled[j, k]
                        new_idx += 1
        return new_batch

    def _inverse_scale(self, continuous_output: torch.Tensor) -> torch.Tensor:
        scaled = torch.zeros((continuous_output.shape[0], self.n_features))
        output = torch.zeros_like(continuous_output)
        continuous_idx = 0
        for i in range(self.n_features):
            if i not in self.categorical_values:
                scaled[:, i] = continuous_output[:, continuous_idx] = continuous_output[:, continuous_idx]
                continuous_idx += 1

        scaled = torch.tensor(self.scalar.inverse_transform(scaled), dtype=torch.float)
        continuous_idx = 0
        for i in range(self.n_features):
            if i not in self.categorical_values:
                output[:, continuous_idx] = scaled[:, i]
                continuous_idx += 1
        return output


