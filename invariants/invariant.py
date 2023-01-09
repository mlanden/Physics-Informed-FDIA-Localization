import torch
from typing import FrozenSet, List
from .predicate import Predicate


class Invariant:
    def __init__(self, antecedent: FrozenSet[Predicate], consequent: FrozenSet[Predicate], support: int,
                 predicate_map: dict):
        self.support = support
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

    def confidence(self, input_states: torch.Tensor, network_outputs: List[torch.Tensor]) -> torch.Tensor:
        if len(input_states.shape) == 3:
            input_states = input_states[:, -1, :]
        product = torch.ones((1, network_outputs[0].shape[0]), device=input_states.device)
        for predicate in self.antecedent_objs:
            confidence = predicate.confidence(input_states, network_outputs)
            product *= confidence

        for predicate in self.consequent_objs:
            confidence = predicate.confidence(input_states, network_outputs)
            product *= confidence
        assert product.shape[0] == 1
        return product

    def _convert_predicates(self):
        for p in self.antecedent:
            self.antecedent_objs.append(self.predicate_map[p])

        for p in self.consequent:
            self.consequent_objs.append(self.predicate_map[p])

    def __str__(self):
        out = "Support: " + str(self.support) + "\n"
        for p in self.antecedent_objs:
            out += str(p) + "\n"
        out += "=>\n"
        for p in self.consequent_objs:
            out += str(p) + "\n"
        return out[:-1]
