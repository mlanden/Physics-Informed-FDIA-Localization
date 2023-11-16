import pandas as pd
import numpy as np
import torch
from typing import List
import matplotlib.pyplot as plt
from .equation import Equation

V_IDX = 3
THETA_IDX = 2
MVAR_IDX = 0
class ReactivePowerEquation(Equation):
    def __init__(self, n_buses, bus_num, bus_type_file) -> None:
        self.n_buses = n_buses
        self.bus_num = bus_num
        self.bus_types = pd.read_csv(bus_type_file)
        self.bus_losses = []
        self.ybus_base = 4 * self.n_buses + 2 * self.n_buses * self.bus_num
    
    def evaluate(self, states):
        k_base_idx = 4 * self.bus_num
        v_k = states[:, k_base_idx + V_IDX]
        power_k = states[:, k_base_idx + MVAR_IDX]
        theta_k = states[:, k_base_idx + THETA_IDX]

        bus_loss = 0
        for bus_j in range(self.n_buses):
            j_base_idx = 4 * bus_j
            v_j = states[:, j_base_idx + V_IDX]
            theta_j = states[:, j_base_idx + THETA_IDX]
            radians = (np.pi / 180) * (theta_k - theta_j)
            admittance_idx = self.ybus_base + 2 * bus_j

            bus_power = v_j * (states[:, admittance_idx] * np.sin(radians)
                                - states[:, admittance_idx + 1] * np.cos(radians))
            bus_loss += bus_power
        bus_loss *= v_k
        bus_loss -= power_k
        self.bus_losses.append(np.abs(bus_loss))
        return bus_loss ** 2
    
    def confidence_loss(self, input_states: torch.Tensor, network_outputs: torch.tensor, targets: torch.Tensor) -> torch.Tensor:
        ybus_base = 2 * self.n_buses + 2 * self.n_buses * self.bus_num
        k_bus_idx = 2 * self.bus_num
        bus_type = self.bus_types.iloc[self.bus_num, 0]
        if bus_type  == 1:
            # PQ
            power_k = input_states[:, k_bus_idx + MVAR_IDX]
            theta_k = network_outputs[:, k_bus_idx]
            v_k = network_outputs[:, k_bus_idx + 1]
        elif bus_type == 2:
            # PV 
            power_k = network_outputs[:, k_bus_idx]
            theta_k = network_outputs[:, k_bus_idx + 1]
            v_k = input_states[:, k_bus_idx + 1]
        elif bus_type == 3:
            # Slack
            power_k = network_outputs[:, k_bus_idx]
            theta_k = input_states[:, k_bus_idx]
            v_k = input_states[:, k_bus_idx + 1]

        bus_loss = 0
        for bus_j in range(self.n_buses):
            j_bus_idx = 2 * bus_j
            bus_type = self.bus_types.iloc[bus_j, 0]
            if bus_type == 1:
                # PQ
                theta_j = network_outputs[:, j_bus_idx]
                v_j = network_outputs[:, j_bus_idx + 1]
            elif bus_type == 2:
                # PV
                theta_j = network_outputs[:, j_bus_idx]
                v_j = input_states[:, j_bus_idx + 1]
            elif bus_type == 3:
                theta_j = input_states[:, j_bus_idx]
                v_j = input_states[:, j_bus_idx + 1]

            radians = (torch.pi / 180) * (theta_k - theta_j)
            admittance_idx = ybus_base + 2 * bus_j
            bus_power = v_j * (input_states[:, admittance_idx] * torch.sin(radians)
                                - input_states[:, admittance_idx + 1] * torch.cos(radians))
            bus_loss += bus_power
        bus_loss *= v_k
        bus_loss -= power_k
        return bus_loss ** 2
    
    def loss_plot(self):
        self.bus_losses = np.concatenate(self.bus_losses)
        print(f"Average bus error, reactive power bus {self.bus_num}:", np.mean(self.bus_losses), np.std(self.bus_losses))
        plt.hist(self.bus_losses)
        plt.xlabel("Reactive Power Error per Bus")
        plt.ylabel("Count")
        plt.savefig("Reactive_power_error.png")
