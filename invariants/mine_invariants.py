from typing import List, Tuple
from collections import defaultdict
import torch
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt

from .predicate import Predicate
from .invariant import Invariant
from datasets import ICSDataset
from utils import cfp_growth, MISTree


def mine_invariants(predicates: List[Predicate], dataset: ICSDataset, conf: dict):
    index_to_predicate, predicate_to_index = create_mappings(predicates)
    predicate_counts, predicates_satisfied = assign_predicates(predicates, dataset)
    features, labels = dataset.get_data()

    min_supports = {}
    local_min_support = conf["train"]["gamma"]
    global_min_support = conf["train"]["theta"]
    max_depth = conf["train"]["max_depth"]

    for i, predicate in enumerate(predicates):
        min_supports[i] = max(local_min_support * predicate_counts[i], len(features) *
                              global_min_support)

    print("Starting to build predicate sets")
    freq_patterns, pattern_counts, tree = cfp_growth(features, predicates_satisfied, min_supports, max_depth)
    lengths = [len(p) for p in freq_patterns]
    fig, ax = plt.subplots()
    ax.hist(lengths)
    plt.tight_layout()
    plt.savefig("predicate_lengths.png")
    print("Predicate sets built")

    closed_sets = find_closed_predicate_sets(freq_patterns, pattern_counts, predicate_counts,
                                             min(min_supports.values()))
    rules = generate_rules(closed_sets, predicate_counts, tree)
    print("Number of rules:", len(rules))
    invariants = create_invariant_objs(rules, index_to_predicate)
    return invariants


def generate_rules(closed_sets, predicate_counts, tree, min_confidence=1):
    rules = []
    for i in range(2, len(closed_sets)):
        for freq_sets in closed_sets[i]:
            predicates = [frozenset([item]) for item in freq_sets]
            if i == 1:
                compute_confidence(freq_sets, predicates, tree, predicate_counts, rules, min_confidence)
            else:
                create_rules(freq_sets, predicates, tree, predicate_counts, rules, min_confidence)
    return rules


def assign_predicates(predicates: List[Predicate], dataset: ICSDataset) -> Tuple[dict, dict]:
    """Determine which states satisfy each predicate"""
    features, labels = dataset.get_data()
    predicate_counts = {}

    # The predicates satisfied in each state
    predicates_satisfied = defaultdict(list)
    for i, predicate in enumerate(predicates):
        total = 0
        for state in features:
            if predicate.is_satisfied(torch.tensor(state)):
                predicates_satisfied[tuple(state)].append(i)
                total += 1
        predicate_counts[i] = total

    return predicate_counts, predicates_satisfied


def find_closed_predicate_sets(predicate_sets, set_counts, predicate_counts, min_support):
    pattern_sizes = defaultdict(list)
    for item in predicate_counts:
        if predicate_counts[item] >= min_support:
            key = frozenset([item])
            set_counts[key] = predicate_counts[item]
            pattern_sizes[0].append(key)

    for pattern in predicate_sets:
        pattern_sizes[len(pattern) - 1].append(frozenset(pattern))

    closed_sets = []
    for i in range(len(pattern_sizes) - 1):
        closed = []
        prev_sets = pattern_sizes[i]
        next_sets = pattern_sizes[i + 1]
        for prev_set in prev_sets:
            valid = True
            for next_set in next_sets:
                if prev_set.issubset(next_set) and set_counts[prev_set] == set_counts[next_set]:
                    valid = False
                    break
            if valid:
                closed.append(prev_set)
        closed_sets.append(closed)
    closed_sets.append(pattern_sizes[-1])
    return closed_sets


def compute_confidence(freq_set, predicates: list, tree: MISTree, pattern_counts: dict, rules: list,
                       min_confidence: float):
    assert len(freq_set) == 2
    rule_sets = []
    for consequence in predicates:
        antecedent = freq_set - consequence
        if antecedent not in pattern_counts:
            pattern_counts[antecedent] = tree.support(list(antecedent))

        confidence = pattern_counts[freq_set] / pattern_counts[antecedent]
        if confidence >= min_confidence:
            rules.append((antecedent, consequence, confidence))
            rule_sets.append(consequence)
        return rule_sets


def create_apriori_sets(sets, k):
    good_sets = []
    length = len(sets)
    for i in range(length):
        for j in range(i + 1, length):
            l1 = list(sets[i])[: k - 2]
            l2 = list(sets[j])[: k - 2]
            l1.sort()
            l2.sort()

            if l1 == l2:
                good_sets.append(sets[i] | sets[j])
    return good_sets


def create_rules(freq_set, predicates: List, tree: MISTree, pattern_counts: dict, rules: list, min_confidence: float):
    k = len(predicates[0])
    if len(freq_set) > k + 1:
        joined_sets = create_apriori_sets(predicates, k)
        rule_sets = compute_confidence(freq_set, joined_sets, tree, pattern_counts, rules, min_confidence)
        if len(rule_sets) > 1:
            create_rules(freq_set, rule_sets, tree, pattern_counts, rules, min_confidence)


def create_invariant_objs(rules, index_to_predicate):
    invariants = []
    for antecedent, consequent, confidence in rules:
        antecedents = frozenset([index_to_predicate[i] for i in antecedent])
        consequents = frozenset([index_to_predicate[i] for i in consequent])
        invariants.append(Invariant(antecedents, consequents))
    return invariants


def create_mappings(predicates):
    index_to_predicate = {}
    predicate_to_index = {}
    for i in range(len(predicates)):
        index_to_predicate[i] = predicates[i]
        predicate_to_index[predicates[i]] = i
    return index_to_predicate, predicate_to_index
