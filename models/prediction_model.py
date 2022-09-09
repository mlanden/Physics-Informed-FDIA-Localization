import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class PredictionModel(nn.Module, ABC):
    @abstractmethod
    def loss(self, predicted: torch.Tensor, target: torch.Tensor, scaled_target: torch.Tensor,
             hidden_states: torch.Tensor = None) -> torch.Tensor:
        pass

    @abstractmethod
    def embed(self, batch: torch.Tensor) -> torch.Tensor:
        pass
