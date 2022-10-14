import torch
import numpy as np

from .predicate import Predicate


class EventPredicate(Predicate):

    def __init__(self, coefficients, bias, target_idx, epsilon, positive_error, continuous_features):
        super().__init__()
        self.positive_error = positive_error
        self.continuous_features = continuous_features
        self.target_idx = target_idx
        self.epsilon = epsilon
        self.bias = bias
        self.coefficients = coefficients

    def is_satisfied(self, state: torch.Tensor) -> bool:
        continuous_idx = 0
        total = self.bias
        for i in range(len(state)):
            if i not in self.continuous_features:
                continue

            if i != self.target_idx:
                total += self.coefficients[continuous_idx] * state[i]
                continuous_idx += 1

        if self.positive_error:
            total += self.epsilon
            return state[self.target_idx] < total
        else:
            total -= self.epsilon
            return state[self.target_idx] > total

    def confidence(self, network_outputs: torch.Tensor) -> torch.Tensor:
        continuous_output = network_outputs[1]
        coef_idx = 0
        target = 0

        total = self.bias
        i = 0
        while i < len(continuous_output):
            if self.continuous_features[i] != self.target_idx:
                total += self.coefficients[coef_idx] * continuous_output[i]
                coef_idx += 1
            else:
                target = continuous_output[i]
            i += 1

        return torch.abs(target - total)

    def __hash__(self):
        if self.hash is None:
            hash_ = []
            hash_.extend(self.coefficients)
            hash_.append(self.bias)
            hash_.append(self.target_idx)
            hash_.append(self.positive_error)
            self.hash = hash(tuple(hash_))
        return self.hash

    def __eq__(self, other):
        if not isinstance(other, EventPredicate):
            return False

        return np.all(self.coefficients == other.coefficients) and self.bias == other.bias and self.target_idx == \
               other.target_idx and self.positive_error == other.positive_error
