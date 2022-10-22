import torch
from typing import FrozenSet
from .predicate import Predicate


class Invariant:
    def __init__(self, antecedent: FrozenSet[Predicate], consequent: FrozenSet[Predicate], predicate_map: dict):
        self.predicate_map = predicate_map
        self.antecedent = antecedent
        self.consequent = consequent
        self.antecedent_objs = []
        self.consequent_objs = []
        self._convert_predicates()

    def satisfied(self, state):
        for predicate in self.antecedent_objs:
            if not predicate.is_satisfied(state):
                return True

        for predicate in self.consequent_objs:
            if not predicate.is_satisfied(state):
                return False
        return True

    def confidence(self, network_outputs: torch.Tensor) -> torch.Tensor:
        product = torch.ones((network_outputs[0].shape[0], 1))
        for predicate in self.antecedent_objs:
            product *= predicate.confidence(network_outputs)

        for predicate in self.consequent_objs:
            product *= predicate.confidence(network_outputs)

        return torch.mean(product)

    def _convert_predicates(self):
        for p in self.antecedent:
            self.antecedent_objs.append(self.predicate_map[p])

        for p in self.consequent:
            self.consequent_objs.append(self.predicate_map[p])
