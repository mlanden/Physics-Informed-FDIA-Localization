import torch

from .predicate import Predicate


class EventPredicate(Predicate):

    def __init__(self, coefficients, bias, target_idx, epsilon, positive_error, continuous_features):
        self.positive_error = positive_error
        self.continuous_features = continuous_features
        self.target_idx = target_idx
        self.epsilon = epsilon
        self.bias = bias
        self.coefficients = coefficients

    def is_satisfied(self, state: torch.Tensor) -> bool:
        total = self._evaluate(state)
        if self.positive_error:
            total += self.epsilon
            return state[self.target_idx] > total
        else:
            total -= self.epsilon
            return state[self.target_idx] < total

    def confidence(self, features: torch.Tensor) -> torch.Tensor:
        predicted = self._evaluate(features)
        return torch.abs(features[self.target_idx] - predicted)

    def _evaluate(self, state: torch.Tensor) -> float:
        total = self.bias
        for i in self.continuous_features:
            if i != self.target_idx:
                total += self.coefficients[i] * state[i]
        return total