import pandas as pd
import numpy as np
import torch
from typing import List
import matplotlib.pyplot as plt
from .equation import Equation

V_IDX = 5
THETA_IDX = 4
GEN_MVAR_IDX = 0
LOAD_MVAR_IDX = 2
class ReactivePowerEquation(Equation):
    def __init__(self, n_buses, admittance_file, base_mva=100) -> None:
        self.n_buses = n_buses
        self.admittance = pd.read_csv(admittance_file).fillna("").applymap(ReactivePowerEquation._to_complex)
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
            load_k = states[k_base_idx+ LOAD_MVAR_IDX]
            generator_k = states[k_base_idx + GEN_MVAR_IDX]
            theta_k = states[k_base_idx + THETA_IDX]

            bus_loss = 0
            for bus_j in range(self.n_buses):
                j_base_idx = 6 * bus_j
                v_j = states[j_base_idx + V_IDX]
                theta_j = states[j_base_idx + THETA_IDX]
                radians = (np.pi / 180) * (theta_k - theta_j)
                bus_power = v_j * (self.admittance.iloc[bus_k, bus_j].real * np.sin(radians)
                                   - self.admittance.iloc[bus_k, bus_j].imag * np.cos(radians))
                bus_loss += bus_power
            bus_loss *= v_k
            bus_loss -= generator_k
            bus_loss += load_k
            self.bus_losses.append(np.abs(bus_loss))
            loss += bus_loss ** 2
        return loss
    
    def confidence_loss(self, input_states: torch.Tensor, network_outputs: torch.tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = 0

        return loss
    
    def loss_plot(self):
        small = np.count_nonzero(np.array(self.bus_losses) < 5)
        print("Fraction small:", small / len(self.bus_losses))
        print("average bus error:", np.mean(self.bus_losses))
        print("Std bus error", np.std(self.bus_losses))
        plt.hist(self.bus_losses)
        plt.xlabel("Real Power Error per Bus")
        plt.ylabel("Count")
        plt.savefig("Reactive_power_error.png")
