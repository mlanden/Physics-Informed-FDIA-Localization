import torch
import numpy as np

from .predicate import Predicate


class EventPredicate(Predicate):

    def __init__(self, model, target_idx, epsilon, positive_error, continuous_features):
        super().__init__()
        self.positive_error = positive_error
        self.continuous_features = continuous_features
        self.target_idx = target_idx
        self.epsilon = epsilon
        self.model = model

        self.features = list(continuous_features)
        self.features.remove(target_idx)

    def is_satisfied(self, states: np.ndarray) -> bool:
        state_features = states[:, self.features]
        pred = self.model.predict(state_features)

        if self.positive_error:
            pred += self.epsilon
            return states[:, self.target_idx] < pred
        else:
            pred -= self.epsilon
            return states[:, self.target_idx] > pred

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
            hash_.extend(tuple(self.model.coef_.flatten()))
            hash_.extend(tuple(self.model.intercept_.flatten()))
            hash_.append(self.target_idx)
            hash_.append(self.positive_error)
            self.hash = hash(tuple(hash_))
        return self.hash

    def __eq__(self, other):
        if not isinstance(other, EventPredicate):
            return False

        return np.all(self.model.coef_ == other.model.coef_) and np.all(self.model.intercept_ == other.model.intercept_)\
               and self.target_idx == other.target_idx and self.positive_error == other.positive_error

    def __str__(self):
        out = "Event Predicate: "
        return out