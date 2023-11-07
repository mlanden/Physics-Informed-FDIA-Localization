import pandas as pd
import numpy as np
import torch
from typing import List
import matplotlib.pyplot as plt
from .equation import Equation

V_IDX = 3
THETA_IDX = 2
MW_IDX = 1
class RealPowerEquation(Equation):

    def __init__(self, n_buses, bus_num, admittance_file, bus_type_file) -> None:
        self.n_buses = n_buses
        self.bus_num = bus_num
        self.admittance = pd.read_csv(admittance_file).fillna("").map(RealPowerEquation._to_complex)
        self.bus_types = pd.read_csv(bus_type_file)
        self.bus_losses = []

    @classmethod
    def _to_complex(cls, s: str):
        if len(s) == 0:
            return complex(0)
        else:
            s = s.replace(" ", "")
            if "j" in s:
                s = s.replace("j", "") + "j"
            s = s.replace("i", "j")
            return complex(s)
        
    def evaluate(self, states):
        k_base_idx = 4 * self.bus_num
        v_k = states[k_base_idx + V_IDX]
        power_k = states[k_base_idx + MW_IDX]
        theta_k = states[k_base_idx + THETA_IDX]

        bus_loss = 0
        for bus_j in range(self.n_buses):
            j_base_idk = 4 * bus_j
            theta_j = states[j_base_idk + THETA_IDX]
            v_j = states[j_base_idk + V_IDX]
            radians = (np.pi / 180) * (theta_k - theta_j)
            bus_power =  v_j * (self.admittance.iloc[self.bus_num, bus_j].real * np.cos(radians)
                                + self.admittance.iloc[self.bus_num, bus_j].imag * np.sin(radians))
            bus_loss += bus_power
        bus_loss *= v_k
        bus_loss -= power_k
        # print(bus_k, bus_loss)
        self.bus_losses.append(np.abs(bus_loss))
        return bus_loss ** 2

        
    def confidence_loss(self, input_states: torch.Tensor, network_outputs: torch.tensor, targets: torch.Tensor) -> torch.Tensor:
        # Inputs : generation, loads
        # Outputs : voltage magnitude and angle
        k_bus_idx = 2 * self.bus_num
        bus_type = self.bus_types.iloc[self.bus_num, 0]
        if bus_type == 1:
            # PQ
            power_k = input_states[:, k_bus_idx+ 1]
            theta_k = network_outputs[:, k_bus_idx]
            v_k = network_outputs[:, k_bus_idx + 1]
        elif bus_type == 2:
            # PV
            power_k = input_states[:, k_bus_idx + 1]
            theta_k = network_outputs[:, k_bus_idx + 1]
            v_k = input_states[:, k_bus_idx + 1]
        elif bus_type == 3:
            power_k = network_outputs[:, k_bus_idx + 1] 
            theta_k = input_states[:, k_bus_idx]
            v_k = input_states[:, k_bus_idx + 1]
        print(v_k)
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
                theta_j = network_outputs[:, j_bus_idx + 1]
                v_j = input_states[:, j_bus_idx + 1]
            elif bus_type == 3:
                theta_j = input_states[:, j_bus_idx]
                v_j = input_states[:, j_bus_idx + 1]
                
            radians = (torch.pi / 180) * (theta_k - theta_j)

            bus_power = torch.abs(v_j) * (self.admittance.iloc[self.bus_num, bus_j].real * torch.cos(radians)
                                            + self.admittance.iloc[self.bus_num, bus_j].imag * torch.sin(radians))
            bus_loss += bus_power
        bus_loss *= torch.abs(v_k)
        bus_loss -= power_k
        return bus_loss ** 2
    
    def loss_plot(self):
        print(f"Average bus error, real power bus {self.bus_num}:", np.mean(self.bus_losses))
        print("Std bus error", np.std(self.bus_losses))
        plt.hist(self.bus_losses)
        plt.xlabel("Real Power Error per Bus")
        plt.ylabel("Count")
        plt.savefig("Real_power_error.png")
