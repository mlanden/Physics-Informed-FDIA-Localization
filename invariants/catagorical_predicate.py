import torch

from .predicate import Predicate


class CategoricalPredicate(Predicate):

    def __init__(self, idx: int, categorical_idx: int, class_value: int):
        self.categorical_idx = categorical_idx
        self.idx = idx
        self.class_value = class_value

        self.loss = torch.nn.CrossEntropyLoss()

    def is_satisfied(self, state: torch.Tensor) -> bool:
        """ Determine whether a state of the system has a categorical variable set to the target class

        Parameters
        -----------
        state : Tensor
        The complete system state
        """
        return state[self.idx] == self.class_value

    def confidence(self, features: torch.Tensor) -> torch.Tensor:
        """ Returns the cross entropy between the given class logits and the target class

        Parameters
        -----------
        features: list[Tensor]
        A list of logit outputs for each categorical variable in the system
        """

        return self.loss(features[self.categorical_idx], self.class_value)
