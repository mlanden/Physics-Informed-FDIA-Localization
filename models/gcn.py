import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.nn as gnn


class GCN(nn.Module):
    def __init__(self, conf, n_inputs, n_outputs, dense=False) -> None:
        super().__init__()
        self.hidden_size = conf["model"]["hidden_size"]
        self.n_heads = conf["model"]["n_heads"]
        dropout = conf["model"]["dropout"]
        self.dense = dense
        self.n_buses = conf["data"]["n_buses"]
        self.conv1 = gnn.GATConv(n_inputs, self.hidden_size, self.n_heads, dropout=dropout)
        self.conv2 = gnn.GATConv(self.hidden_size * self.n_heads, n_outputs,
                                  self.n_heads, concat=False, dropout=dropout)
        if self.dense:
            self.classify = nn.Linear(n_outputs * self.n_buses, 2 * self.n_buses)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        h = self.conv1(x, edge_index, edge_attr)
        h = h.relu()
        h = self.conv2(h, edge_index, edge_attr)
        if self.dense:
            h = h.view(-1, self.classify.in_features)
            logits = self.classify(h)
            return logits
        return h
