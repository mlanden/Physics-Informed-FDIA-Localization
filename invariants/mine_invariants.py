from typing import List, Tuple
from collections import defaultdict
import torch

from .predicate import Predicate

from datasets import ICSDataset
from utils import MISTree

def mine_invariants(predicates: List[Predicate], dataset: ICSDataset, conf: dict):
    predicate_counts, predicates_satisfied = assign_predicates(predicates, dataset)
    features, labels = dataset.get_data()

    min_supports = {}
    local_min_support = conf["train"]["gamma"]
    global_min_support = conf["train"]["theta"]
    for predicate in predicates:
        min_supports[predicate] = max(local_min_support * predicate_counts[predicate], len(features) *
                                      global_min_support)

    mis_tree = MISTree()
    mis_tree.build(features, predicates_satisfied, min_supports)
    print("MIS Tree built")


def assign_predicates(predicates: List[Predicate], dataset: ICSDataset) -> Tuple[dict, dict]:
    """Determine which states satisfy each predicate"""
    features, labels = dataset.get_data()
    predicate_counts = {}

    # The predicates satisfied in each state
    predicates_satisfied = defaultdict(list)
    for predicate in predicates:
        total = 0
        for i, state in enumerate(features):
            if predicate.is_satisfied(torch.tensor(state)):
                predicates_satisfied[tuple(state)].append(predicate)
                total += 1
        predicate_counts[predicate] = total

    return predicate_counts, predicates_satisfied


