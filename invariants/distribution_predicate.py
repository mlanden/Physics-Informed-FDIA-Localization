import numpy as np
import torch

from .predicate import Predicate


class DistributionPredicate(Predicate):

    def __init__(self, means, variances, weights, state_idx, continuous_idx, distribution_idx):
        super().__init__()
        self.continuous_idx = continuous_idx
        self.means = means
        self.variances = variances
        self.distributions = [torch.distributions.Normal(means[i], variances[i]) for i in range(len(means))]
        self.weights = weights
        self.state_idx = state_idx
        self.distribution_idx = distribution_idx

    def is_satisfied(self, state: torch.Tensor) -> bool:
        membership_probabilities = torch.zeros(len(self.weights))
        for i in range(len(self.distributions)):
            log_prob = self.distributions[i].log_prob(state[self.state_idx])
            membership_probabilities[i] = self.weights[i] * log_prob
        return torch.argmax(membership_probabilities) == self.distribution_idx

    def confidence(self, network_outputs: torch.Tensor) -> torch.Tensor:
        continuous_outputs = network_outputs[1]
        total = torch.zeros(1)
        for i in range(len(self.distributions)):
            log_prob = self.distributions[i].log_prob(continuous_outputs[self.continuous_idx])
            total += self.weights[i] * log_prob

        log_prob = self.distributions[self.distribution_idx].log_prob(continuous_outputs[self.continuous_idx])
        return self.weights[self.distribution_idx] * log_prob / total

    def __hash__(self):
        if self.hash is None:
            hash_ = []
            hash_.extend(self.means)
            hash_.extend(self.variances)
            hash_.extend(self.weights)
            hash_.append(self.state_idx)
            hash_.append(self.distribution_idx)
            self.hash = hash(tuple(hash_))
        return self.hash

    def __eq__(self, other):
        if not isinstance(other, DistributionPredicate):
            return False

        return np.all(self.means == other.means) and np.all(self.variances == other.variances) and \
               np.all(self.weights == other.weights) and np.all(self.state_idx == other.state_idx) and \
               np.all(self.distribution_idx == other.distribution_idx)
