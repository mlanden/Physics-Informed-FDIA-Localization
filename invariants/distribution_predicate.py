import torch

from .predicate import Predicate


class DistributionPredicate(Predicate):

    def __init__(self, means, variances, weights, state_idx, distribution_idx):
        self.distributions = [torch.distributions.Normal(means[i], variances[i]) for i in range(len(means))]
        self.weights = weights
        self.idx = idx
        self.state_idx = state_idx
        self.distribution_idx = distribution_idx

    def is_satisfied(self, state: torch.Tensor) -> bool:
        membership_probabilities = torch.zeros(len(self.weights))
        for i in range(len(self.distributions)):
            log_prob = self.distributions[i].log_prob(state[self.state_idx])
            membership_probabilities[i] = self.weights[i] * log_prob
        return torch.argmax(membership_probabilities) == self.distribution_idx

    def confidence(self, features: torch.Tensor) -> torch.Tensor:
        pass
