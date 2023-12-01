import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.nn as gnn


class GCN(nn.Module):
    def __init__(self, conf, n_inputs, n_outputs) -> None:
        super().__init__()
        self.hidden_size = conf["model"]["hidden_size"]
        self.n_heads = conf["model"]["n_heads"]
        self.conv1 = gnn.GATConv(n_inputs, self.hidden_size, self.n_heads)
        self.conv2 = gnn.GATConv(self.hidden_size * self.n_heads, self.hidden_size,
                                  self.n_heads, concat=False)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        h = self.conv1(x, edge_index, edge_attr)
        h = h.relu()
        h = self.conv2(h, edge_index, edge_attr)
        return h
