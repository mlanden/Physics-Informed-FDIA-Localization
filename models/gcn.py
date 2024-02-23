import torch.nn as nn
import torch_geometric.nn as gnn


class GCN(nn.Module):
    def __init__(self, conf, n_inputs, n_outputs) -> None:
        super().__init__()
        hidden_size = conf["model"]["hidden_size"]
        n_heads = conf["model"]["n_heads"]
        dropout = conf["model"]["dropout"]
        n_layers = conf["model"]["n_layers"]
        self.n_buses = conf["data"]["n_buses"]
        self.n_outputs = n_outputs
        self.gnns = nn.ModuleList()
        for i in range(n_layers):
            # if i < n_layers - 1:
            #     concat = True
            # else:
            #     concat = dense
            self.gnns.append(gnn.GATConv(n_inputs if i == 0 else hidden_size * n_heads,
                                         hidden_size,
                                         n_heads, dropout=dropout))
        self.pinn_conv = gnn.GATConv(hidden_size * n_heads, n_outputs, n_heads, concat=False, dropout=dropout)
        self.localize_conv = gnn.GATConv(hidden_size * n_heads, hidden_size, n_heads, concat=False, dropout=dropout)
        self.classify = nn.Linear(hidden_size * self.n_buses, 2 * self.n_buses)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        for i, layer in enumerate(self.gnns):
            x = layer(x, edge_index, edge_attr)
            x = x.relu()

        pinn_output = self.pinn_conv(x, edge_index, edge_attr)
        localization_outut = self.localize_conv(x, edge_index, edge_attr)
        localization_outut = localization_outut.relu()
        localization_outut = localization_outut.view(len(data), -1)
        localization_outut = self.classify(localization_outut)
        return pinn_output, localization_outut
