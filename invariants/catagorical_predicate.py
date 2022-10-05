import torch

from .predicate import Predicate


class CategoricalPredicate(Predicate):

    def __init__(self, idx: int, categorical_idx: int, class_value: int):
        super().__init__()
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

    def confidence(self, network_outputs: torch.Tensor) -> torch.Tensor:
        """ Returns the cross entropy between the given class logits and the target class

        Parameters
        -----------
        network_outputs: list[Tensor]
        A list of logit outputs for each categorical variable in the system
        """
        categorical_outputs = network_outputs[0]

        return self.loss(categorical_outputs[self.categorical_idx], self.class_value)

    def __hash__(self):
        if self.hash is None:
            self.hash = hash((self.idx, self.class_value))
        return self.hash

    def __eq__(self, other):
        if not isinstance(other, CategoricalPredicate):
            return False

        return self.idx == other.idx and self.class_value == other.class_value
