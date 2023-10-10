from typing import List
from abc import ABC, abstractmethod
import torch


class Equation(ABC):

    @abstractmethod
    def evaluate(self, states):
        pass

    @abstractmethod
    def confidence_loss(self, input_states: torch.Tensor, network_outputs: List[torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        pass
