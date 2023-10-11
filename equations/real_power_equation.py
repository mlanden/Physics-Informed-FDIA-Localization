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

    def __init__(self, n_buses, admittance_file) -> None:
        self.n_buses = n_buses
        self.admittance = pd.read_csv(admittance_file).fillna("").applymap(RealPowerEquation._to_complex)
        self.bus_losses = []

    @classmethod
    def _to_complex(cls, s: str):
        if len(s) == 0:
            return complex(0)
        else:
            s = s.replace(" ", "")
            if "j" in s:
                s = s.replace("j", "") + "j"
            return complex(s)
        
    def evaluate(self, states):
        loss = 0
        for bus_k in range(self.n_buses):
            k_base_idx = 6 * bus_k
            v_k = states[-1, k_base_idx + V_IDX]
            load_mw_k = states[-1, k_base_idx + LOAD_MW_IDX]
            generator_mw_k = states[-1, k_base_idx + GEN_MW_IDX]
            theta_k = states[-1, k_base_idx + THETA_IDX]
            # print(k_base_idx, v_k, load_mw_k, generator_mw_k, theta_k)
            bus_loss = 0
            for bus_j in range(self.n_buses):
                j_base_idk = 6 * bus_j
                v_j = states[-1, j_base_idk + V_IDX]
                theta_j = states[-1, j_base_idk + THETA_IDX]

                bus_power =  v_j * (self.admittance.iloc[bus_k, bus_j].real * np.cos(theta_k - theta_j)
                                                                    + self.admittance.iloc[bus_k, bus_j].imag * np.sin(theta_k - theta_j))
                bus_loss += bus_power
            bus_loss *= v_k
            bus_loss -= generator_mw_k
            bus_loss += load_mw_k
            # print(bus_k, bus_loss)
            self.bus_losses.append(bus_loss)
            loss += (bus_loss ** 2)
        # print(loss)
        return loss            
        
    def confidence_loss(self, input_states: torch.Tensor, network_outputs: List[torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        loss = 0
        output = network_outputs[0]
        for bus_k in range(self.n_buses):
            k_bus_idx = 6 * bus_k
            v_k = input_states[:, -1, k_bus_idx + V_IDX] + output[:, k_bus_idx + V_IDX]
            load_mw_k = input_states[:, -1, k_bus_idx + LOAD_MW_IDX] + output[:, k_bus_idx + LOAD_MW_IDX]
            generator_mw_k = input_states[:, -1, k_bus_idx + GEN_MW_IDX] + output[:, k_bus_idx + GEN_MW_IDX]
            theta_k = input_states[:, -1, k_bus_idx  + THETA_IDX] + output[:, k_bus_idx + THETA_IDX]

            bus_loss = 0
            for bus_j in range(self.n_buses):
                j_bus_idx = 6 * bus_j
                v_j = input_states[:, -1, j_bus_idx + V_IDX] + output[:, j_bus_idx + V_IDX]
                theta_j = input_states[:, -1, j_bus_idx + THETA_IDX] + output[:, j_bus_idx + THETA_IDX]

                bus_power = torch.abs(v_j) * (self.admittance.iloc[bus_k, bus_j].real * torch.cos(theta_k - theta_j)
                                               + self.admittance.iloc[bus_k, bus_j].imag * torch.sin(theta_k - theta_j))
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
        plt.hist(self.bus_losses)
        plt.xlabel("Real Power Error per Bus")
        plt.ylabel("Count")
        plt.savefig("Real_power_error.png")
