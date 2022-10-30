import numpy as np
import torch

from .predicate import Predicate


class DistributionPredicate(Predicate):

    def __init__(self, model, threshold, state_idx, continuous_idx, distribution_idx):
        super().__init__()
        self.threshold = threshold
        self.continuous_idx = continuous_idx
        self.model = model
        self.distributions = [torch.distributions.Normal(model.means_[i, 0], model.covariances_[i, 0, 0])
                              for i in range(model.means_.shape[0])]
        self.weights = model.weights_
        self.state_idx = state_idx
        self.distribution_idx = distribution_idx

    def is_satisfied(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states)
        states = states[:, self.state_idx].reshape(-1, 1)
        diffs = states[1:] - states[:-1]
        score = self.model.score_samples(diffs)
        cluster = self.model.predict(diffs)
        return np.hstack((False, np.logical_and(score >= self.threshold, cluster == self.distribution_idx)))

    def confidence(self, input_states, network_outputs: torch.Tensor) -> torch.Tensor:
        continuous_outputs = network_outputs[0]
        if len(continuous_outputs.shape) == 1:
            continuous_outputs = continuous_outputs.view(1, -1)
        total = torch.zeros(continuous_outputs.shape[0])
        for i in range(len(self.distributions)):
            log_prob = self.distributions[i].log_prob(continuous_outputs[:, self.continuous_idx])
            total += self.weights[i] * log_prob

        log_prob = self.distributions[self.distribution_idx].log_prob(continuous_outputs[:, self.continuous_idx])
        confidence = torch.div(self.weights[self.distribution_idx] * log_prob, total)
        return confidence.view(-1, 1)

    def __hash__(self):
        if self.hash is None:
            hash_ = []
            hash_.extend(tuple(self.model.means_.flatten()))
            hash_.extend(tuple(self.model.covariances_.flatten()))
            hash_.extend(tuple(self.weights.flatten()))
            hash_.append(self.state_idx)
            hash_.append(self.distribution_idx)
            self.hash = hash(tuple(hash_))
        return self.hash

    def __eq__(self, other):
        if not isinstance(other, DistributionPredicate):
            return False

        return np.all(self.model.means_ == other.model.means_) and np.all(self.model.covariances_ ==
            other.model.covariances_) and np.all(self.model.weights_ == other.model.weights_) and np.all(self.state_idx == other.state_idx) and \
               np.all(self.distribution_idx == other.distribution_idx)

    def __str__(self):
        out = "Distribution:"
        return out
