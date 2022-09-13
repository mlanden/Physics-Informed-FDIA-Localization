import joblib
from typing import Tuple, Union
from os import path

import torch
import torch.nn as nn
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
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
        self.n_features = conf["data"]["n_features"]
        self.embedding_size = conf["model"]["embedding_size"]

        self.activation = activations[conf["model"]["activation"]]
        self.input_linear = nn.Linear(self.n_features + len(CATEGORICAL_VALUES) * (self.embedding_size - 1),
                                      hidden_layers[0])
        self.embeddings = nn.ModuleList()
        self.classifications = nn.ModuleList()
        for size in CATEGORICAL_VALUES.values():
            self.embeddings.append(nn.Embedding(size, self.embedding_size))
            self.classifications.append(nn.Linear(hidden_layers[-1], size))

        self.rnns = nn.ModuleList()
        for i in range(len(hidden_layers[:-1])):
            self.rnns.append(nn.LSTM(hidden_layers[i], hidden_layers[i + 1], batch_first=True))

        self.output_linear = nn.Linear(hidden_layers[-1], self.n_features - len(CATEGORICAL_VALUES))

    def forward(self, x, hidden_states=None) -> Union[Tuple[torch.Tensor, Tuple], torch.Tensor]:
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

        if hidden_states is not None:
            return outputs, hidden_states
        else:
            return outputs

    def scale(self, target):
        if self.scalar is None:
            self.scalar = joblib.load(self.scalar_path)
        scaled_target = torch.tensor(self.scalar.transform(target), dtype=torch.float)
        return scaled_target

    def predict(self, batch: torch.Tensor, hidden_states: torch.Tensor = None) -> Union[
        Tuple[torch.Tensor, Tuple], torch.Tensor]:
        if hidden_states is not None:
            intermediate, hidden_states = self.forward(batch, hidden_states)
        else:
            intermediate = self.forward(batch)

        output = torch.zeros(batch.shape[1:])
        continuous_outputs = self.output_linear(intermediate)
        continuous_outputs = self._inverse_scale(continuous_outputs)
        continuous_idx = 0
        classification_idx = 0
        for i in range(batch.shape[-1]):
            if i in CATEGORICAL_VALUES:
                # Cross entropy_loss
                logits = self.classifications[classification_idx](intermediate)
                class_ = torch.argmax(logits, dim=1)
                output[:, i] = class_
                classification_idx += 1
            else:
                # MSE loss
                output[:, i] = continuous_outputs[:, continuous_idx]
                continuous_idx += 1

        if hidden_states is not None:
            return output, hidden_states
        else:
            return output

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

    def _inverse_scale(self, continuous_output: torch.Tensor) -> torch.Tensor:
        scaled = torch.zeros((continuous_output.shape[0], self.n_features))
        output = torch.zeros_like(continuous_output)
        continuous_idx = 0
        for i in range(self.n_features):
            if i not in CATEGORICAL_VALUES:
                scaled[:, i] = continuous_output[:, continuous_idx] = continuous_output[:, continuous_idx]
                continuous_idx += 1

        scaled = torch.tensor(self.scalar.inverse_transform(scaled), dtype=torch.float)
        continuous_idx = 0
        for i in range(self.n_features):
            if i not in CATEGORICAL_VALUES:
                output[:, continuous_idx] = scaled[:, i]
                continuous_idx += 1
        return output


def swat_loss(model, batch: torch.Tensor, target: torch.Tensor, hidden_states=None) -> Union[
    Tuple[torch.Tensor, Tuple], torch.Tensor]:
    if isinstance(model, DDP):
        scaled_target = model.module.scale(target)
    else:
        scaled_target = model.scale(target)

    if hidden_states is not None:
        outputs, hidden_states = model.forward(batch, hidden_states)
    else:
        outputs = model.forward(batch)

    loss = torch.tensor(0.0)
    continuous_idx = 0
    classification_idx = 0
    class_loss = nn.CrossEntropyLoss()
    continuous_loss = nn.MSELoss()
    for i in range(batch.shape[-1]):
        if i in CATEGORICAL_VALUES:
            # Cross entropy_loss
            logits = outputs[classification_idx + 1]
            target_class = target[:, i].long()
            if CATEGORICAL_VALUES[i] == 2:
                target_class[:] -= 1

            loss += class_loss(logits, target_class)
            classification_idx += 1
        else:
            # MSE loss
            predicted = outputs[0][:, continuous_idx]
            loss += continuous_loss(predicted, scaled_target[:, i])
            continuous_idx += 1

    if hidden_states is not None:
        return loss, hidden_states
    else:
        return loss
