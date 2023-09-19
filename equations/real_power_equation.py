import pandas as pd
import numpy as np
import torch
from typing import List

from .equation import Equation

class RealPowerEquation(Equation):

    def __init__(self, n_buses, admittance_file) -> None:
        self.n_buses = n_buses
        self.admittance = pd.read_csv(admittance_file).fillna("").applymap(RealPowerEquation._to_complex)

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
            k_base_idx = 5 * bus_k
            v_k = states[-1, k_base_idx + 4]
            power_k = states[-1, k_base_idx + 2]
            theta_k = states[-1, k_base_idx + 3]
            actual_power = 0
            for bus_j in range(self.n_buses):
                j_base_idk = 5 * bus_j
                v_j = states[-1, j_base_idk + 4]
                theta_j = states[-1, j_base_idk + 3]

                bus_power = np.abs(v_k) * np.abs(v_j) * (self.admittance.iloc[bus_k, bus_j].real * np.cos(theta_k - theta_j)
                                                                    + self.admittance.iloc[bus_k, bus_j].imag * np.sin(theta_k - theta_j))
                actual_power += bus_power
            loss += (power_k - actual_power) ** 2
            print(loss, power_k, actual_power)
        return loss            
        
    def confidence_loss(self, input_states: torch.Tensor, network_outputs: List[torch.Tensor]) -> torch.Tensor:
        return 0
    