import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch


class CNN(nn.Module):
    def __init__(self, conf, n_inputs, n_outputs) -> None:
        super().__init__()

        hidden_size = conf["model"]["hidden_size"]
        self.dropout = conf["model"]["dropout"]
        n_layers = conf["model"]["n_layers"]
        k = conf["model"]["k"]

        self.shared_layers = nn.ModuleList()
        linear_input = n_inputs
        for i in range(n_layers):
            linear_input -= k - 1
            self.shared_layers.append(nn.Conv1d(1 if i == 0 else hidden_size,
                                                hidden_size, k))
        
        self.localization_layers = nn.ModuleList()
        for i in range(n_layers):
            linear_input -= k - 1
            self.localization_layers.append(nn.Conv1d(hidden_size, hidden_size, k))
        self.localization_linear = nn.Linear(hidden_size * linear_input, n_outputs)
        
        self.pinn_layers = nn.ModuleList()
        for i in range(n_layers):
            self.pinn_layers.append(nn.Conv1d(hidden_size, hidden_size, k))
        self.pinn_linear = nn.Linear(hidden_size * linear_input, n_outputs)
    
    def forward(self, inputs, targets):
        input = inputs.unsqueeze(1)
        attack = torch.argwhere((torch.max(targets, dim=1).values == 0)).flatten()
        pinn_output = input[attack]
        localization_output = input

        for layer in self.shared_layers:
            pinn_output = layer(pinn_output)
            pinn_output = F.dropout(pinn_output, self.dropout)
            pinn_output = pinn_output.relu()
            localization_output = layer(localization_output)
            localization_output = F.dropout(localization_output, self.dropout)
            localization_output = localization_output.relu()

        for layer in self.localization_layers:
            localization_output = layer(localization_output)
            localization_output = F.dropout(localization_output, self.dropout)
            localization_output = localization_output.relu()
        localization_output = torch.flatten(localization_output, 1)
        localization_output = self.localization_linear(localization_output)
        
        for layer in self.pinn_layers:
            pinn_output = layer(pinn_output)
            pinn_output = F.dropout(pinn_output, self.dropout)
            pinn_output = pinn_output.relu()
        pinn_output = torch.flatten(pinn_output, 1)
        pinn_output = self.pinn_linear(pinn_output)
        
        return pinn_output, localization_output, input[attack].squeeze(1)
