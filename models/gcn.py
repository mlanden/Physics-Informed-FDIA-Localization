import torch.nn as nn
import torch_geometric.nn as gnn


class GCN(nn.Module):
    def __init__(self, conf, n_inputs, n_outputs, dense=False) -> None:
        super().__init__()
        hidden_size = conf["model"]["hidden_size"]
        n_heads = conf["model"]["n_heads"]
        dropout = conf["model"]["dropout"]
        n_layers = conf["model"]["n_layers"]
        self.dense = dense
        n_buses = conf["data"]["n_buses"]
        self.gnns = nn.ModuleList()
        for i in range(n_layers):
            # if i < n_layers - 1:
            #     concat = True
            # else:
            #     concat = dense
            concat = i < n_layers - 1
            self.gnns.append(gnn.GATConv(n_inputs if i == 0 else hidden_size * n_heads,
                                         hidden_size if i < n_layers - 1 else n_outputs,
                                         n_heads, dropout=dropout, concat=concat))
        if self.dense:
            self.classify = nn.Linear(n_outputs * n_buses, 2 * n_buses)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        for i, layer in enumerate(self.gnns):
            x = layer(x, edge_index, edge_attr)
            if i < len(self.gnns) - 1:
                x = x.relu()
        if self.dense:
            x = x.view(-1, self.classify.in_features)
            logits = self.classify(x)
            return logits
        return x
