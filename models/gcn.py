import torch
import torch.nn as nn
import torch_geometric.nn as gnn


class GCN(nn.Module):
    def __init__(self, conf, n_inputs, n_outputs) -> None:
        super().__init__()
        hidden_size = conf["model"]["hidden_size"]
        n_heads = conf["model"]["n_heads"]
        dropout = conf["model"]["dropout"]
        n_layers = conf["model"]["n_layers"]
        n_stacks = conf["model"]["n_stacks"]
        n_iters = conf["model"]["n_iters"]
        self.n_buses = conf["data"]["n_buses"]
        self.n_outputs = n_outputs

        self.gnns = nn.ModuleList()
        for i in range(n_layers):
            self.gnns.append(gnn.ARMAConv(n_inputs if i == 0 else hidden_size,
                                         hidden_size, n_stacks, n_iters, dropout=dropout))
        
        self.pinn_conv = nn.ModuleList()
        for i in range(n_layers):
            self.pinn_conv.append(gnn.ARMAConv(hidden_size, hidden_size if i < n_layers - 1 else n_outputs, 
                                              n_stacks, n_iters, dropout=dropout))
            
        self.localize_conv = nn.ModuleList()
        for i in range(n_layers):
            self.localize_conv.append(gnn.ARMAConv(hidden_size + n_inputs if i == 0 else hidden_size, hidden_size,
                                                   n_stacks, n_iters, dropout=dropout))
        self.classify = nn.Linear(hidden_size * self.n_buses, 2 * self.n_buses)

    def forward(self, data):
        x = data.x
        inputs = x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        edge_weights = torch.abs(data.edge_attr[:, 0])

        for i, layer in enumerate(self.gnns):
            x = layer(x, edge_index, edge_weights)
            x = x.relu()
        
        pinn_output = x
        for i, layer in enumerate(self.pinn_conv):
            pinn_output = layer(pinn_output, edge_index, edge_weights)
            if i < len(self.pinn_conv) - 1:
                pinn_output = pinn_output.relu()

        localization_outut = torch.hstack((x, inputs))
        for i, layer in enumerate(self.localize_conv):
            localization_outut = layer(localization_outut, edge_index, edge_weights)
            localization_outut = localization_outut.relu()

        localization_outut = localization_outut.view(len(data), -1)
        localization_outut = self.classify(localization_outut)
        return pinn_output, localization_outut
