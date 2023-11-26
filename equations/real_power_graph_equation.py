import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

from .equation import Equation


V_IDX = 3
THETA_IDX = 2
MW_IDX = 1
class RealPowerGraphEquation(Equation):
    def __init__(self, n_buses, bus_num, bus_type_file) -> None:
        self.n_buses = n_buses
        self.bus_num = bus_num
        self.bus_types = pd.read_csv(bus_type_file)
        self.bus_type = self.bus_types.iloc[bus_num, 0]
        self.bus_losses = []

    def evaluate(self, graph: Data):
        n_nodes = len(graph.x)
        n_graphs = n_nodes // self.n_buses
        node_idxs = torch.tensor(np.array(list(range(self.bus_num, n_nodes, self.n_buses))))
        
        if self.bus_type == 1:
            v_k = graph.y[node_idxs, 1]
            power_k = graph.x[node_idxs, 1]
            theta_k = graph.y[node_idxs, 0]
        elif self.bus_type == 2:
            v_k = graph.x[node_idxs, 1]
            power_k = graph.x[node_idxs, 0]
            theta_k = graph.y[node_idxs, 1]
        elif self.bus_type == 3:
            v_k = graph.x[node_idxs, 1]
            power_k = graph.y[node_idxs, 1]
            theta_k = graph.x[node_idxs, 0]

        gs = []
        bs = []
        v_j = []
        theta_j = []
        for i in node_idxs:
            edge_indexes = (graph.edge_index[0, :] == i).nonzero().view(-1)
            targets = graph.edge_index[1, edge_indexes].view(n_graphs, -1)
            print(targets)

            gs.append(graph.edge_attr[edge_indexes, 0].view(1, -1))
            bs.append(graph.edge_attr[edge_indexes, 1].view(1, -1))
            v = []
            theta = []
            for j in targets:
                bus_type = self.bus_types.iloc[j.item() % self.n_buses, 0]
                if bus_type == 1:
                    theta.append(graph.y[j, 0].view(1, 1))
                    v.append(graph.y[j, 1].view(1, 1))
                elif bus_type == 2:
                    theta.append(graph.y[j, 1].view(1, 1))
                    v.append(graph.x[j, 1].view(1, 1))
                elif bus_type == 3:
                    theta.append(graph.x[j, 0].view(1, 1))
                    v.append(graph.x[j, 1].view(1, 1))
            theta = torch.cat(theta).t()
            v = torch.cat(v).t()
            theta_j.append(theta)
            v_j.append(v)
            
        bus_loss = self._compute_bus_loss(graph, node_idxs, v_k, power_k, 
                                          theta_k, gs, bs, v_j, theta_j)
        quit()
        return bus_loss ** 2

    def confidence_loss(self, input_graph: Data, output: torch.tensor, targets: torch.Tensor) -> torch.Tensor:
        n_nodes = len(input_graph.x)
        node_idxs = torch.tensor(np.array(list(range(self.bus_num, n_nodes, self.n_buses))))

        if self.bus_type == 1:
            v_k = output[node_idxs, 1]
            power_k = input_graph.x[node_idxs, 1]
            theta_k = output[node_idxs, 0]
        elif self.bus_type == 2:
            v_k = input_graph.x[node_idxs, 1]
            power_k = input_graph.x[node_idxs, 0]
            theta_k = output[node_idxs, 1]
        elif self.bus_type == 3:
            v_k = input_graph.x[node_idxs, 1]
            power_k = output[node_idxs, 1]
            theta_k = input_graph.x[node_idxs, 0]

        gs = []
        bs = []
        v_j = []
        theta_j = []
        for i in node_idxs:
            edge_indexes = (input_graph.edge_index[0, :] == i).nonzero().view(-1)
            targets = input_graph.edge_index[1, edge_indexes]
            gs.append(input_graph.edge_attr[edge_indexes, 0].view(1, -1))
            bs.append(input_graph.edge_attr[edge_indexes, 1].view(1, -1))
            v = []
            theta = []
            for j in targets:
                bus_type = self.bus_types.iloc[j.item() % self.n_buses, 0]
                if bus_type == 1:
                    theta.append(output[j, 0].view(1, 1))
                    v.append(output[j, 1].view(1, 1))
                elif bus_type == 2:
                    theta.append(output[j, 1].view(1, 1))
                    v.append(input_graph.x[j, 1].view(1, 1))
                elif bus_type == 3:
                    theta.append(input_graph.x[j, 0].view(1, 1))
                    v.append(input_graph.x[j, 1].view(1, 1))
            theta = torch.cat(theta).t()
            v = torch.cat(v).t()
            theta_j.append(theta)
            v_j.append(v)

        bus_loss = self._compute_bus_loss(input_graph, node_idxs, v_k, power_k, theta_k,
                                          gs, bs, v_j, theta_j)
        return bus_loss ** 2
    
    def _compute_bus_loss(self, graph, node_idxs, v_k, power_k, theta_k, gs, bs, v_j, theta_j):
        theta_j = torch.cat(theta_j, dim=0)
        v_j = torch.cat(v_j, dim=0)
        gs = torch.cat(gs, dim=0)
        bs = torch.cat(bs, dim=0)

        theta_k = theta_k.unsqueeze(1)
        theta_j = torch.cat([theta_j, theta_k], dim=1)
        v_k_ = v_k.unsqueeze(1)
        v_j = torch.cat([v_j, v_k_], dim=1)
        theta_k = theta_k.expand(-1, theta_j.size(1))
        node_g = graph.x[node_idxs, -2].unsqueeze(1)
        node_b = graph.x[node_idxs, -1].unsqueeze(1)
        gs = torch.cat([gs, node_g], dim=1)
        bs = torch.cat([bs, node_b], dim=1)

        radians = (torch.pi / 180) * (theta_k - theta_j)
        bus_loss = v_j * (gs * torch.cos(radians) + bs * torch.sin(radians))
        bus_loss = torch.sum(bus_loss, dim=1)
        bus_loss *= v_k
        bus_loss -= power_k
        return bus_loss