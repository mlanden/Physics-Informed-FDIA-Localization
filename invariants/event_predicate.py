import torch
import numpy as np
from typing import List

from .predicate import Predicate


class EventPredicate(Predicate):

    def __init__(self, model, target_idx, epsilon, positive_error, continuous_features):
        super().__init__()
        self.positive_error = positive_error
        self.continuous_features = continuous_features
        self.target_idx = target_idx
        self.epsilon = epsilon
        self.model = model
        self.linear_model = torch.nn.Linear(len(self.model.coef_), 1)
        with torch.no_grad():
            self.linear_model.weight = torch.nn.Parameter(torch.as_tensor(self.model.coef_).view(1, -1), requires_grad=False)
            self.linear_model.bias = torch.nn.Parameter(torch.as_tensor(self.model.intercept_), requires_grad=False)

        self.features = list(continuous_features)
        self.features.remove(target_idx)

    def is_satisfied(self, states, network_outputs=None) -> bool:
        if network_outputs is None:
            state_features = states[:, self.features]

            pred = self.model.predict(state_features)

            if self.positive_error:
                pred += self.epsilon
                return states[:, self.target_idx] < pred
            else:
                pred -= self.epsilon
                return states[:, self.target_idx] > pred
        else:
            confidence = self.confidence(states, network_outputs)
            # print(confidence < self.epsilon)
            return confidence < self.epsilon

    def confidence(self, input_states, network_outputs: List[torch.Tensor]) -> torch.Tensor:
        self.linear_model.to(input_states.device)

        continuous_output = network_outputs[0]
        input_ = torch.zeros((input_states.shape[0], self.linear_model.in_features), device=input_states.device)
        input_idx = 0
        target = 0
        for i, feature_idx in enumerate(self.continuous_features):
            if len(input_states.shape) == 3:
                state_prediction = input_states[:, feature_idx] + continuous_output[:, i]
            else:
                state_prediction = input_states[:, feature_idx] + continuous_output[:, i]

            if feature_idx != self.target_idx:
                input_[:, input_idx] = state_prediction
                input_idx += 1
            else:
                target = state_prediction.view(-1, 1)

        predicted = self.linear_model(input_)
        confidence = torch.transpose(torch.abs(target - predicted) / target, 0, 1)
        return confidence

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
        out = f"Event Predicate: Target: {self.target_idx}"
        return out