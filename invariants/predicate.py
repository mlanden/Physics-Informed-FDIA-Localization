from typing import List
from abc import ABC, abstractmethod
import torch
import numpy as np


class Predicate(ABC):

    def __init__(self):
        self.hash = None

    @abstractmethod
    def is_satisfied(self, states: np.ndarray) -> np.ndarray:
        """ Return whether the predicate is satisfied in a state

        Parameters
        ---------
        states: Tensor
        The state of the system to test
        """
        pass

    @abstractmethod
    def confidence(self, input_states: torch.Tensor, network_outputs: List[torch.Tensor]) -> torch.Tensor:
        """Returns the confidence that a state satisfies the predicate 
        
        Parameters
        -------------
        input_states: Tensor
        The input sequences to the network

        network_outputs: Tensor
        The state of the system to test 
        """
        pass

    @abstractmethod
    def __hash__(self) -> int:
        pass

    @abstractmethod
    def __eq__(self, other):
        pass