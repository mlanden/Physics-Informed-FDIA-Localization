import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class PredictionModel(nn.Module, ABC):

    @abstractmethod
    def embed(self, batch: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def predict(self, batch: torch.Tensor, hidden_states: torch.Tensor = None) -> torch.Tensor:
        pass
