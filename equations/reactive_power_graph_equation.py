import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

from .equation import Equation


class ReactivePowerGrapghEquation(Equation):
    def __init__(self, n_buses, bus_num) -> None:
        self.n_buses = n_buses
        self.bus_num = bus_num

    def evaluate(self, graph: Data):
        n_nodes = len(graph.x)
        node_idxs = torch.tensor(np.array(list(range(self.bus_num, n_nodes, self.n_buses))))

        power_k = graph.x[node_idxs, 0]
        theta_k = graph.y[node_idxs, 0]
        v_k = graph.y[node_idxs, 1]

        edge_indexes = [(graph.edge_index[0, :] == i).nonzero().view(1, -1) for i in node_idxs]
        edge_indexes = torch.cat(edge_indexes, dim=0)
        bus_loss = 0
        for j in range(edge_indexes.size(1)):
            targets = graph.edge_index[1, edge_indexes[:, j]]
            gs = graph.edge_attr[edge_indexes[:, j], 0]
            bs = graph.edge_attr[edge_indexes[:, j], 1]
            theta_j = graph.y[targets, 0]
            v_j = graph.y[targets, 1]
            
            radians = (torch.pi / 180) * (theta_k - theta_j)
            bus_loss += v_j * (gs * torch.sin(radians) - bs * torch.cos(radians))
        bus_loss *= v_k
        bus_loss -= power_k
        return bus_loss ** 2

    def confidence_loss(self, input_graph: Data, output: torch.tensor, targets: torch.Tensor) -> torch.Tensor:
        n_nodes = len(input_graph.x)
        node_idxs = torch.tensor(np.array(list(range(self.bus_num, n_nodes, self.n_buses))))

        power_k = input_graph.x[node_idxs, 0]
        theta_k = output[node_idxs, 0]
        v_k = output[node_idxs, 1]

        edge_indexes = [(input_graph.edge_index[0, :] == i).nonzero().view(1, -1) for i in node_idxs]
        edge_indexes = torch.cat(edge_indexes, dim=0)
        bus_loss = 0
        for j in range(edge_indexes.size(1)):
            targets = input_graph.edge_index[1, edge_indexes[:, j]]
            gs = input_graph.edge_attr[edge_indexes[:, j], 0]
            bs = input_graph.edge_attr[edge_indexes[:, j], 1]
            theta_j = output[targets, 0]
            v_j = output[targets, 1]

            radians = (torch.pi / 180) * (theta_k - theta_j)
            bus_loss += v_j * (gs * torch.sin(radians) - bs * torch.cos(radians))
        bus_loss *= v_k
        bus_loss -= power_k
        return bus_loss ** 2
