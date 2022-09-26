from typing import List

import numpy as np
from sklearn.mixture import GaussianMixture

from datasets import ICSDataset
from .predicate import Predicate
from .catagorical_predicate import CategoricalPredicate
from .distribution_predicate import DistributionPredicate


def generate_predicates(dataset: ICSDataset, conf: dict) -> List[Predicate]:
    predicates = []
    categorical_values = dataset.get_categorical_features()
    for idx, n_classes in categorical_values.items():
        for val in range(n_classes):
            predicates.append(CategoricalPredicate(idx, val))

    distribution_predicates = _generate_distribution_predicates(dataset, conf)
    predicates.extend(distribution_predicates)

    return predicates


def _generate_distribution_predicates(dataset: ICSDataset, conf: dict) -> List[Predicate]:
    features, labels = dataset.get_data()
    categorical_values = dataset.get_categorical_features()
    deltas = features[1:, ...] - features[:-1, ...]

    predicates = []
    max_components = conf["train"]["max_gmm_components"]
    for feature in range(features.shape[-1]):
        if feature in categorical_values:
            continue
        train_data = deltas[:, feature].reshape(-1, 1)
        max_score = -np.Inf
        best_model = None
        for k in range(1, max_components + 1):
            gmm = GaussianMixture(n_components=k)
            print(f"Fitting model for feature {feature} with k={k}")
            gmm.fit(train_data)
            bic = gmm.bic(train_data)

            if bic > max_score:
                max_score = bic
                best_model = gmm

        means = best_model.means_.flatten()
        variances = best_model.covariances_.flatten()
        weights = best_model.weights_
        for i in range(len(means)):
            predicates.append(DistributionPredicate(means, variances, weights, feature, i))

    return predicates


# def _generate_event_predicates