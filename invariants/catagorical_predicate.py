import torch

from .predicate import Predicate


class CategoricalPredicate(Predicate):

    def __init__(self, idx: int, class_value: int):
        self.idx = idx
        self.class_value = class_value

        self.loss = torch.nn.CrossEntropyLoss()

    def is_satisfied(self, state: torch.Tensor) -> bool:
        return state[self.idx] == self.class_value

    def confidence(self, features: torch.Tensor) -> torch.Tensor:
        return self.loss(features, self.class_value)
