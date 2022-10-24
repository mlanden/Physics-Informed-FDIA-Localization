import numpy as np
import torch

from .predicate import Predicate


class CategoricalPredicate(Predicate):

    def __init__(self, idx: int, categorical_idx: int, class_value: int):
        super().__init__()
        self.categorical_idx = categorical_idx
        self.idx = idx
        self.class_value = class_value

        self.loss = torch.nn.CrossEntropyLoss()

    def is_satisfied(self, states: np.ndarray) -> bool:
        """ Determine whether a state of the system has a categorical variable set to the target class

        Parameters
        -----------
        states : Tensor
        The complete system state
        """
        return states[:, self.idx] == self.class_value

    def confidence(self, input_states, network_outputs: torch.Tensor) -> torch.Tensor:
        """ Returns the cross entropy between the given class logits and the target class

        Parameters
        -----------
        network_outputs: list[Tensor]
        A list of logit outputs for each categorical variable in the system
        """
        categorical_output = network_outputs[self.categorical_idx + 1]
        target = torch.full((categorical_output.shape[0],), self.class_value)
        return self.loss(categorical_output, target)

    def __hash__(self):
        if self.hash is None:
            self.hash = hash((self.idx, self.class_value))
        return self.hash

    def __eq__(self, other):
        if not isinstance(other, CategoricalPredicate):
            return False

        return self.idx == other.idx and self.class_value == other.class_value

    def __str__(self):
        out = f"Categorical: Index:{self.idx}, Value: {self.class_value}"
        return out
