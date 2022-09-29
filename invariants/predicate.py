from abc import ABC, abstractmethod
import torch


class Predicate(ABC):

    def __init__(self):
        self.hash = None

    @abstractmethod
    def is_satisfied(self, state: torch.Tensor) -> bool:
        """ Return whether the predicate is satisfied in a state

        Parameters
        ---------
        state: Tensor
        The state of the system to test
        """
        pass

    @abstractmethod
    def confidence(self, network_outputs: torch.Tensor) -> torch.Tensor:
        """Returns the confidence that a state satisfies the predicate 
        
        Parameters
        -------------
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