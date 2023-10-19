import pandas as pd
import numpy as np
import torch
from typing import List
import matplotlib.pyplot as plt
from .equation import Equation

V_IDX = 5
THETA_IDX = 4
GEN_MW_IDX = 1
LOAD_MW_IDX = 3
class RealPowerEquation(Equation):

    def __init__(self, n_buses, admittance_file, base_mva=100) -> None:
        self.n_buses = n_buses
        self.admittance = pd.read_csv(admittance_file).fillna("").applymap(RealPowerEquation._to_complex)
        self.base_mva = base_mva
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
        loss = 0
        for bus_k in range(self.n_buses):
            k_base_idx = 6 * bus_k
            v_k = states[k_base_idx + V_IDX]
            load_k = states[k_base_idx + LOAD_MW_IDX]
            generator_k = states[k_base_idx + GEN_MW_IDX]
            theta_k = states[k_base_idx + THETA_IDX]

            bus_loss = 0
            for bus_j in range(self.n_buses):
                j_base_idk = 6 * bus_j
                v_j = states[j_base_idk + V_IDX]
                theta_j = states[j_base_idk + THETA_IDX]
                radians = (np.pi / 180) * (theta_k - theta_j)
                bus_power =  v_j * (self.admittance.iloc[bus_k, bus_j].real * np.cos(radians)
                                    + self.admittance.iloc[bus_k, bus_j].imag * np.sin(radians))
                bus_loss += bus_power
            bus_loss *= v_k
            bus_loss -= generator_k
            bus_loss += load_k
            # print(bus_k, bus_loss)
            self.bus_losses.append(np.abs(bus_loss))
            loss += (bus_loss ** 2)
        # print(loss)
        return loss            
        
    def confidence_loss(self, input_states: torch.Tensor, network_outputs: torch.tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = 0
        # Inputs : generation, loads
        # Outputs : voltage magnitude and angle
        for bus_k in range(self.n_buses):
            k_bus_inputs = 4 * bus_k
            k_bus_outputs = 2 * bus_k
            load_mw_k = input_states[:, k_bus_inputs + LOAD_MW_IDX]
            generator_mw_k = input_states[:, k_bus_inputs + GEN_MW_IDX]
            theta_k = network_outputs[:, k_bus_outputs]
            v_k = network_outputs[:, k_bus_outputs + 1]

            bus_loss = 0
            for bus_j in range(self.n_buses):
                j_bus_idx = 2 * bus_j
                v_j = network_outputs[:, j_bus_idx + 1]
                theta_j = network_outputs[:, j_bus_idx]
                radians = (torch.pi / 180) * (theta_k - theta_j)

                bus_power = torch.abs(v_j) * (self.admittance.iloc[bus_k, bus_j].real * torch.cos(radians)
                                               + self.admittance.iloc[bus_k, bus_j].imag * torch.sin(radians))
                bus_loss += bus_power
            bus_loss *= torch.abs(v_k)
            bus_loss -= generator_mw_k
            bus_loss += load_mw_k
            loss += (bus_loss ** 2)
        return loss
    
    def loss_plot(self):
        small = np.count_nonzero(np.array(self.bus_losses) < 5)
        print("Fraction small:", small / len(self.bus_losses))
        print("average bus error:", np.mean(self.bus_losses))
        print("Std bus error", np.std(self.bus_losses))
        plt.hist(self.bus_losses)
        plt.xlabel("Real Power Error per Bus")
        plt.ylabel("Count")
        plt.savefig("Real_power_error.png")
