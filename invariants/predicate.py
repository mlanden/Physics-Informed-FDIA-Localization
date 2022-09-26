from abc import ABC, abstractmethod
import torch


class Predicate(ABC):

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
    def confidence(self, features: torch.Tensor) -> torch.Tensor:
        """Returns the confidence that a state satisfies the predicate 
        
        Parameters
        -------------
        features: Tensor
        The state of the system to test 
        """
        pass
