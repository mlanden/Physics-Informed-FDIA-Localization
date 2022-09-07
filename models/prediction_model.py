import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class PredictionModel(nn.Module, ABC):
    @abstractmethod
    def loss(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def embed(self, batch: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def reverse_embed(self, batch: torch.Tensor) -> torch.Tensor:
        pass
