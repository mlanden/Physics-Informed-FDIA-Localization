import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

from .equation import Equation


class ReactivePowerGrapghEquation(Equation):
    def __init__(self, n_buses, bus_num, bus_type_file) -> None:
        self.n_buses = n_buses
        self.bus_num = bus_num
        self.bus_types = pd.read_csv(bus_type_file)
        self.bus_type = self.bus_types.iloc[bus_num, 0]

    def evaluate(self, graph: Data):
        n_nodes = len(graph.x)
        node_idxs = torch.tensor(np.array(list(range(self.bus_num, n_nodes, self.n_buses))))

        if self.bus_type == 1:
            power_k = graph.x[node_idxs, 0]
            theta_k = graph.y[node_idxs, 0]
            v_k = graph.y[node_idxs, 1]
        elif self.bus_type == 2:
            power_k = graph.y[node_idxs, 0]
            theta_k = graph.y[node_idxs, 1]
            v_k = graph.x[node_idxs, 1]
        elif self.bus_type == 3:
            power_k = graph.y[node_idxs, 0]
            theta_k = graph.x[node_idxs, 0]
            v_k = graph.x[node_idxs, 1]
        
        edge_indexes = [(graph.edge_index[0, :] == i).nonzero().view(1, -1) for i in node_idxs]
        edge_indexes = torch.cat(edge_indexes, dim=0)
        bus_loss = 0
        for j in range(edge_indexes.size(1)):
            targets = graph.edge_index[1, edge_indexes[:, j]]
            gs = graph.edge_attr[edge_indexes[:, j], 0]
            bs = graph.edge_attr[edge_indexes[:, j], 1]
            bus_type = self.bus_types.iloc[targets[0].item(), 0]
            if bus_type == 1:
                theta_j = graph.y[targets, 0]
                v_j = graph.y[targets, 1]
            elif bus_type == 2:
                theta_j = graph.y[targets, 1]
                v_j = graph.x[targets, 1]
            elif bus_type == 3:
                theta_j = graph.x[targets, 0]
                v_j = graph.x[targets, 1]
            radians = (torch.pi / 180) * (theta_k - theta_j)
            bus_loss += v_j * (gs * torch.sin(radians) - bs * torch.cos(radians))
            
        gs = graph.x[node_idxs, -2]
        bs = graph.x[node_idxs, -1]
        radians = (torch.pi / 180) * (theta_k - theta_k)
        bus_loss += v_k * (gs * torch.sin(radians) - bs * torch.cos(radians))
        bus_loss *= v_k
        bus_loss -= power_k
        return bus_loss ** 2

    def confidence_loss(self, input_graph: Data, output: torch.tensor, targets: torch.Tensor) -> torch.Tensor:
        n_nodes = len(input_graph.x)
        node_idxs = torch.tensor(np.array(list(range(self.bus_num, n_nodes, self.n_buses))))

        if self.bus_type == 1:
            power_k = input_graph.x[node_idxs, 0]
            theta_k = output[node_idxs, 0]
            v_k = output[node_idxs, 1]
        elif self.bus_type == 2:
            power_k = output[node_idxs, 0]
            theta_k = output[node_idxs, 1]
            v_k = input_graph.x[node_idxs, 1]
        elif self.bus_type == 3:
            power_k = output[node_idxs, 0]
            theta_k = input_graph.x[node_idxs, 0]
            v_k = input_graph.x[node_idxs, 1]

        edge_indexes = [(input_graph.edge_index[0, :] == i).nonzero().view(1, -1) for i in node_idxs]
        edge_indexes = torch.cat(edge_indexes, dim=0)
        bus_loss = 0
        for j in range(edge_indexes.size(1)):
            targets = input_graph.edge_index[1, edge_indexes[:, j]]
            gs = input_graph.edge_attr[edge_indexes[:, j], 0]
            bs = input_graph.edge_attr[edge_indexes[:, j], 1]
            bus_type = self.bus_types.iloc[targets[0].item(), 0]
            if bus_type == 1:
                theta_j = output[targets, 0]
                v_j = output[targets, 1]
            elif bus_type == 2:
                theta_j = output[targets, 1]
                v_j = input_graph.x[targets, 1]
            elif bus_type == 3:
                theta_j = input_graph.x[targets, 0]
                v_j = input_graph.x[targets, 1]
            radians = (torch.pi / 180) * (theta_k - theta_j)
            bus_loss += v_j * (gs * torch.sin(radians) - bs * torch.cos(radians))

        gs = input_graph.x[node_idxs, -2]
        bs = input_graph.x[node_idxs, -1]
        radians = (torch.pi / 180) * (theta_k - theta_k)
        bus_loss += v_k * (gs * torch.sin(radians) - bs * torch.cos(radians))
        bus_loss *= v_k
        bus_loss -= power_k
        return bus_loss ** 2
