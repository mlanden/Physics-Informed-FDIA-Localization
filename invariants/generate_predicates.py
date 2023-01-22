from typing import List
from  os import path
import numpy as np
import pickle
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import Lasso

from datasets import ICSDataset
from .predicate import Predicate
from .catagorical_predicate import CategoricalPredicate
from .distribution_predicate import DistributionPredicate
from .event_predicate import EventPredicate


def generate_predicates(dataset: ICSDataset, conf: dict) -> List[Predicate]:
    load_checkpoint = conf["train"]["load_checkpoint"]
    checkpoint = conf["train"]["checkpoint"]
    checkpoint = path.join("checkpoint", checkpoint, "predicates.pkl")

    if load_checkpoint and path.exists(checkpoint):
        with open(checkpoint, "rb") as fd:
            predicates = pickle.load(fd)
    else:
        predicates = []
        categorical_values = dataset.get_categorical_features()
        for i, (idx, n_classes) in enumerate(categorical_values.items()):
            for val in range(n_classes):
                predicates.append(CategoricalPredicate(idx, i, val))
        print(f"Categorical predicates: {len(predicates)}")

        event_predicates = _generate_event_predicates(dataset, conf)
        predicates.extend(event_predicates)
        distribution_predicates = _generate_distribution_predicates(dataset, conf)
        predicates.extend(distribution_predicates)

        print(f"Distribution predicates: {len(distribution_predicates)}")
        print(f"Event predicates: {len(event_predicates)}")
        print(f"Generated {len(predicates)} predicates. Saved to {checkpoint}")

        with open(checkpoint, "wb") as fd:
            pickle.dump(predicates, fd)

    return predicates


def _generate_distribution_predicates(dataset: ICSDataset, conf: dict) -> List[Predicate]:
    features, labels = dataset.get_data()
    categorical_values = dataset.get_categorical_features()
    deltas = features[1:, ...] - features[:-1, ...]

    predicates = []
    max_components = conf["train"]["max_gmm_components"]
    distribution_threshold = conf["train"]["distribution_threshold"]
    continuous_idx = 0
    for feature in range(features.shape[-1]):
        if feature in categorical_values:
            continue

        print(f"Fitting models for feature {feature}")
        train_data = deltas[:, feature].reshape(-1, 1)

        best_score = np.Inf
        best_model = None
        n_components = -1
        for k in range(1, max_components + 1):
            gmm = GaussianMixture(n_components=k)
            gmm.fit(train_data)
            bic = gmm.bic(train_data)

            if bic < best_score:
                best_score = bic
                best_model = gmm
                n_components = k

        scores = best_model.score_samples(train_data)
        threshold = scores.mean() * distribution_threshold

        for i in range(n_components):
            predicates.append(DistributionPredicate(best_model, threshold, feature, continuous_idx, i))
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
                    if np.max(error) < epsilon:
                        plus_predicate = EventPredicate(model, target_feature, epsilon, True,
                                                        list(continuous_features))
                        predicates.append(plus_predicate)
                        neg_predicate = EventPredicate(model, target_feature, epsilon, False,
                                                       list(continuous_features))
                        predicates.append(neg_predicate)
    return predicates
