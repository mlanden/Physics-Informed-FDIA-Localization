from typing import List

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import Lasso

from datasets import ICSDataset
from .predicate import Predicate
from .catagorical_predicate import CategoricalPredicate
from .distribution_predicate import DistributionPredicate
from .event_predicate import EventPredicate


def generate_predicates(dataset: ICSDataset, conf: dict) -> List[Predicate]:
    predicates = []
    categorical_values = dataset.get_categorical_features()
    for i, (idx, n_classes) in enumerate(categorical_values.items()):
        for val in range(n_classes):
            predicates.append(CategoricalPredicate(idx, i, val))
    print(f"Categorical predicates: {len(predicates)}")

    distribution_predicates = _generate_distribution_predicates(dataset, conf)
    predicates.extend(distribution_predicates)

    event_predicates = _generate_event_predicates(dataset, conf)
    predicates.append(event_predicates)

    print(f"Distribution predicates: {len(distribution_predicates)}")
    print(f"Event predicates: {len(event_predicates)}")
    print(f"Generated {len(predicates)} predicates")

    return predicates


def _generate_distribution_predicates(dataset: ICSDataset, conf: dict) -> List[Predicate]:
    features, labels = dataset.get_data()
    categorical_values = dataset.get_categorical_features()
    deltas = features[1:, ...] - features[:-1, ...]

    predicates = []
    max_components = conf["train"]["max_gmm_components"]
    continuous_idx = 0
    for feature in range(features.shape[-1]):
        if feature in categorical_values:
            continue

        print(f"Fitting models for feature {feature}")
        train_data = deltas[:, feature].reshape(-1, 1)
        max_score = -np.Inf
        best_model = None
        for k in range(1, max_components + 1):
            gmm = GaussianMixture(n_components=k)
            gmm.fit(train_data)
            bic = gmm.bic(train_data)

            if bic > max_score:
                max_score = bic
                best_model = gmm

        means = best_model.means_.flatten()
        variances = best_model.covariances_.flatten()
        weights = best_model.weights_
        for i in range(len(means)):
            predicates.append(DistributionPredicate(means, variances, weights, continuous_idx, i))
        continuous_idx += 1

    return predicates


def _generate_event_predicates(dataset: ICSDataset, conf: dict) -> List[Predicate]:
    epsilon = conf["train"]["event_predicate_error"]
    features, labels = dataset.get_data()
    pre_states = features[:-1, :]
    categorical_values = dataset.get_categorical_features()
    continuous_features = set(range(features.shape[-1])) - set(categorical_values.keys())
    predicates = []

    for actuator in categorical_values:
        pre_value = features[:-1, actuator]
        post_state = features[1:, actuator]

        changes = np.argwhere(pre_value != post_state)
        events = {(pre_value[i].item(), post_state[i].item()) for i in changes}
        for event in events:
            states = pre_states[(pre_value == event[0]) & (post_state == event[1]), :]
            for target_feature in continuous_features:
                active_features = list(continuous_features)
                active_features.remove(target_feature)

                x = states[:, active_features]
                y = states[:, target_feature]

                if len(y) > 5:
                    model = Lasso()
                    model.fit(x, y)

                    y_pred = model.predict(x)
                    error = np.abs(y - y_pred)
                    # print(f"Actuator: {actuator}, target feature: {target_feature}, support: {len(y)},"
                    #       f" error: {np.max(error)}")
                    if np.max(error) < epsilon:
                        plus_predicate = EventPredicate(model.coef_, model.intercept_, target_feature, epsilon, True,
                                                        continuous_features)
                        predicates.append(plus_predicate)
                        neg_predicate = EventPredicate(model.coef_, model.intercept_, target_feature, epsilon, False,
                                                       continuous_features)
                        predicates.append(neg_predicate)
    return predicates
