import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.data import Batch
from torch_geometric.transforms import LaplacianLambdaMax


class GCN(nn.Module):
    def __init__(self, conf, n_inputs, n_outputs) -> None:
        super().__init__()
        hidden_size = conf["model"]["hidden_size"]
        dropout = conf["model"]["dropout"]
        n_heads = conf["model"]["n_heads"]
        n_layers = conf["model"]["n_layers"]
        n_stacks = conf["model"]["n_stacks"]
        n_iters = conf["model"]["n_iters"]
        k = conf["model"]["k"]
        normalization = conf["model"]["noralization"]
        self.n_buses = conf["data"]["n_buses"]
        self.laplacian_max = LaplacianLambdaMax(normalization)
        self.n_outputs = n_outputs

        self.gnns = nn.ModuleList()
        for i in range(n_layers):
            self.gnns.append(gnn.ARMAConv(n_inputs if i == 0 else hidden_size,
                                         hidden_size, n_stacks, n_iters, dropout=dropout))
        
        self.pinn_conv = nn.ModuleList()
        for i in range(n_layers):
            self.pinn_conv.append(gnn.ChebConv(hidden_size, hidden_size, 
                                              k, normalization, dropout=dropout))
        # self.pinn_output = nn.Linear(hidden_size * self.n_buses, n_outputs * self.n_buses)

        self.localize_conv = nn.ModuleList()
        for i in range(n_layers):
            self.localize_conv.append(gnn.ARMAConv(hidden_size + n_inputs if i == 0 else hidden_size, hidden_size,
                                                   n_stacks, n_iters, dropout=dropout))
        self.classify = nn.Linear(hidden_size * self.n_buses, 2 * self.n_buses)

    def forward(self, data, targets):
        data = self.laplacian_max(data)
        localization_outut = data.x
        inputs = data.x
        loc_edge_index = data.edge_index
        loc_edge_weights = torch.sqrt(data.edge_attr[:, 0] ** 2 + data.edge_attr[:, 1] ** 2)

        attack = torch.argwhere((torch.max(targets, dim=1).values == 0))
        no_attack_graphs = data.index_select(attack)
        pinn_data = Batch.from_data_list(no_attack_graphs)
        pinn_data = self.laplacian_max(pinn_data)
        pinn_output = pinn_data.x
        pinn_edge_index = pinn_data.edge_index
        pinn_edge_weights = torch.sqrt(pinn_data.edge_attr[:, 0] ** 2 + pinn_data.edge_attr[:, 1] ** 2)

        for i, layer in enumerate(self.gnns):
            localization_outut = layer(localization_outut, loc_edge_index, loc_edge_weights)#, lambda_max=data.lambda_max)
            localization_outut = localization_outut.relu()
            pinn_output = layer(pinn_output, pinn_edge_index, pinn_edge_weights)#, lambda_max=pinn_data.lambda_max)
            pinn_output = pinn_output.relu()
        
        localization_outut = torch.hstack((localization_outut, inputs))
        for i, layer in enumerate(self.localize_conv):
            localization_outut = layer(localization_outut, loc_edge_index, loc_edge_weights)
            localization_outut = localization_outut.relu()

        localization_outut = localization_outut.view(len(data), -1)
        localization_outut = self.classify(localization_outut)

        for i, layer in enumerate(self.pinn_conv):
            pinn_output = layer(pinn_output, pinn_edge_index, pinn_edge_weights, lambda_max=pinn_data.lambda_max)
            pinn_output = pinn_output.relu()
        
        # pinn_output = pinn_output.view(len(pinn_data), -1)
        # pinn_output = self.pinn_output(pinn_output)
        # pinn_output = pinn_output.view(-1, self.n_outputs)

        return pinn_output, localization_outut, pinn_data
