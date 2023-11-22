import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.nn as gnn


class GCN(nn.Module):
    def __init__(self, conf, n_inputs, n_outputs) -> None:
        super().__init__()
        self.hidden_size = conf["model"]["hidden_size"]
        self.conv1 = gnn.GCNConv(n_inputs, self.hidden_size)
        self.conv2 = gnn.GCNConv(self.hidden_size, 2)

    def forward(self, ins):
        x, edge_index = ins
        h = self.conv1(x, edge_index)
        h = h.relu()
        h = self.conv2(h, edge_index)
        return h
