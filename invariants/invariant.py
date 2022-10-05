import torch
from typing import FrozenSet
from .predicate import Predicate


class Invariant:
    def __init__(self, antecedent: FrozenSet[Predicate], consequent: FrozenSet[Predicate]):
        self.antecedent = antecedent
        self.consequent = consequent

    def evaluate(self, state):
        for predicate in self.antecedent:
            if not predicate.is_satisfied(state):
                return True

        for predicate in self.consequent:
            if not predicate.is_satisfied(state):
                return False
        return True

    def confidence(self, network_outputs: torch.Tensor) -> torch.Tensor:
        product = torch.tensor(1)
        for predicate in self.antecedent:
            product *= predicate.confidence(network_outputs)

        for predicate in self.consequent:
            product *= predicate.confidence(network_outputs)

        return product
