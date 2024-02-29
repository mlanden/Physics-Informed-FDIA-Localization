import numpy as np
import torch
from torch_geometric.data import Data

from .equation import Equation


class PowerGraphEquation(Equation):

    def __init__(self, n_buses) -> None:
        self.n_buses = n_buses

    def evaluate(self, graph: Data):
        lines = graph.edge_index
        voltage_diffs = graph.y[lines[0, :], 0] - graph.y[lines[1, :], 0]
        radians = (torch.pi /180) * voltage_diffs
        cos_voltages = torch.cos(radians)
        sin_voltages = torch.sin(radians)
        gs = graph.edge_attr[:, 0]
        bs = graph.edge_attr[:, 1]
        line_real = gs * cos_voltages + bs * sin_voltages
        line_reactive = gs * sin_voltages - bs * cos_voltages
        line_real *= graph.y[lines[1, :], 1]
        line_reactive *= graph.y[lines[1, :], 1]

        real = torch.empty((len(graph.x)))
        reactive = torch.empty_like(real)
        for i in range(len(graph.x)):
            real[i] = torch.sum(line_real[lines[0, :] == i])
            reactive[i] = torch.sum(line_reactive[lines[0, :] == i])
        
        real *= graph.y[:, 1]
        reactive *= graph.y[:, 1]
        real -= graph.x[:, 1]
        reactive -= graph.x[:, 0]

        loss = real + reactive
        return loss
    
    def confidence_loss(self, input_graph: Data, output: torch.tensor, targets: torch.Tensor) -> torch.Tensor:
        lines = input_graph.edge_index
        voltage_diffs = output[lines[0, :] , 0] - output[lines[1, :], 0]
        radians = (torch.pi / 180) * voltage_diffs
        cos_voltages = torch.cos(radians)
        sin_voltages = torch.sin(radians)
        gs = input_graph.edge_attr[:, 0]
        bs = input_graph.edge_attr[:, 1]
        line_real = gs * cos_voltages + bs * sin_voltages
        line_reative = gs * sin_voltages - bs * cos_voltages
        line_real *= output[lines[1, :], 1]
        line_reative *= output[lines[1, :], 1]

        real = [torch.sum(line_real[lines[0, :] == i]) for i in range(len(input_graph.x))]
        reactive = [torch.sum(line_reative[lines[0, :] == i]) for i in range(len(input_graph.x))]
        real = torch.vstack(real).view(-1)
        reactive = torch.vstack(reactive).view(-1)
        
        real *= output[:, 1]
        reactive *= output[:, 1]
        real -= input_graph.x[:, 1]
        reactive -= input_graph.x[:, 0]

        loss = real ** 2 + reactive ** 2
        return loss

