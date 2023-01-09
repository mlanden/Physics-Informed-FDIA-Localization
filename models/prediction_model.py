
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union

from utils import activations


class PredictionModel(nn.Module):

    def __init__(self, conf, categorical_values):
        super().__init__()
        self.checkpoint = conf["train"]["checkpoint"]
        self.categorical_values = categorical_values

        hidden_layers = conf["model"]["hidden_layers"]
        self.n_features = conf["data"]["n_features"]
        self.embedding_size = conf["model"]["embedding_size"]

        self.activation = activations[conf["model"]["activation"]]
        if self.embedding_size > 0:
            self.input_len = self.n_features + len(self.categorical_values) * (self.embedding_size - 1)
        else:
            self.input_len = self.n_features + sum(self.categorical_values.values()) - len(self.categorical_values)

        self.input_linear = nn.Linear(self.input_len, hidden_layers[0])
        self.embeddings = nn.ModuleList()
        self.classifications = nn.ModuleList()
        for size in self.categorical_values.values():
            if self.embedding_size > 0:
                self.embeddings.append(nn.Embedding(size, self.embedding_size))
            self.classifications.append(nn.Linear(hidden_layers[-1], size))

        self.hidden_size = hidden_layers[1]
        self.rnns = nn.ModuleList()
        for i in range(len(hidden_layers[:-1])):
            self.rnns.append(nn.LSTMCell(hidden_layers[i], hidden_layers[i + 1]))

        self.output_linear = nn.Linear(hidden_layers[-1], self.n_features - len(self.categorical_values))

    def forward(self, unscaled_seq, scaled_seq, hidden_states=None):
        x = self.embed(unscaled_seq, scaled_seq)
        x = self.input_linear(x)
        x = self.activation(x)

        if hidden_states is None:
            # Batch_size,
            hx = torch.zeros((x.shape[0], self.hidden_size), dtype=torch.float32, device=x.device)
            cx = torch.zeros((x.shape[0], self.hidden_size), dtype=torch.float32, device=x.device)
            hidden_states = (hx, cx)

        for time_step in range(x.shape[1]):
            input_ = x[:, time_step, :]
            hidden_states = self.rnns[0](input_, hidden_states)
            for i in range(1, len(self.rnns)):
                hidden_states = self.rnns[i](hidden_states[0], hidden_states)
        output = hidden_states[0]

        continuous_outputs = self.output_linear(output)
        outputs = [continuous_outputs]
        for layer in self.classifications:
            out = layer(output)
            outputs.append(out)

        return outputs, hidden_states

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

    def embed(self, unscaled_seq: torch.Tensor, scaled_seq: torch.Tensor) -> torch.Tensor:
        new_batch = torch.zeros((scaled_seq.shape[0], scaled_seq.shape[1], self.input_len), device=scaled_seq.device)
        for i in range(scaled_seq.shape[0]):
            for j in range(scaled_seq.shape[1]):
                new_idx = 0
                embed_idx = 0
                for feature in range(scaled_seq.shape[-1]):
                    if feature in self.categorical_values:
                        in_ = unscaled_seq[i, j, feature]
                        # swat specific
                        if self.categorical_values[feature] == 2:
                            in_ -= 1

                        if self.embedding_size > 0:
                            embedding = self.embeddings[embed_idx](in_.int())
                        else:
                            embedding = F.one_hot(in_, num_classes=self.categorical_values[feature])
                        new_batch[i, j, new_idx: new_idx + embedding.shape[-1]] = embedding
                        embed_idx += 1
                        new_idx += embedding.shape[-1]
                    else:
                        new_batch[i, j, new_idx] = scaled_seq[i, j, feature]
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
