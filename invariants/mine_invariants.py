import json
import time
import queue
from typing import List, Tuple
from collections import defaultdict
import multiprocessing as mp

import numpy as np
import torch
import pickle
from os import path
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt

from .predicate import Predicate
from .invariant import Invariant
from datasets import ICSDataset
from utils import cfp_growth, MISTree


def mine_invariants(dataset: ICSDataset, conf: dict):
    mp.set_start_method("spawn", force=True)
    checkpoint = conf["train"]["checkpoint"]
    load_checkpoint = conf["train"]["load_checkpoint"]
    predicate_path = path.join("checkpoint", checkpoint, "predicates.pkl")
    if not path.exists(predicate_path):
        raise RuntimeError("No predicates found, create them first")
    n_workers = conf["train"]["n_workers"]

    with open(predicate_path, "rb") as fd:
        predicates = pickle.load(fd)
    print(f"Loaded {len(predicates)} predicates")

    index_to_predicate, predicate_to_index = create_mappings(predicates)
    assign_path = path.join("checkpoint", checkpoint, "assigned_predicates.pkl")
    count_path = path.join("checkpoint", checkpoint, "count_predicates.pkl")

    if path.exists(assign_path) and load_checkpoint:
        with open(assign_path, "rb") as fd:
            predicates_satisfied = pickle.load(fd)
        with open(count_path, "rb") as fd:
            predicate_counts = pickle.load(fd)
    else:
        predicate_counts, predicates_satisfied = assign_predicates(predicates, dataset, n_workers)
        with open(assign_path, "wb") as fd:
            pickle.dump(predicates_satisfied, fd)
        with open(count_path, "wb") as fd:
            pickle.dump(predicate_counts, fd)

    print("Predicates assigned")
    fig, ax = plt.subplots()
    ax.hist(predicate_counts.values(), bins=25)
    plt.savefig("predicates.png")

    # for p in predicate_counts:
    #     print(type(index_to_predicate[p]).__name__, predicate_counts[p])
    # quit()
    features, labels = dataset.get_data()

    min_supports = {}
    local_min_support = conf["train"]["gamma"]
    global_min_support = conf["train"]["theta"]
    max_depth = conf["train"]["max_depth"]
    max_sets = conf["train"]["max_sets"]

    for i, predicate in enumerate(predicates):
        min_supports[i] = max(local_min_support * predicate_counts[i], len(features) *
                              global_min_support)
    print("Mean min support", np.mean(list(min_supports.values())))

    sets_path = path.join("checkpoint", checkpoint, "predicate_sets.pkl")
    counts_path = path.join("checkpoint", checkpoint, "predicate_counts.pkl")
    tree_path = path.join("checkpoint", checkpoint, "tree.pkl")

    if path.exists(sets_path) and load_checkpoint and False:
        with open(sets_path, "rb") as fd:
            closed_sets = pickle.load(fd)
        with open(counts_path, "rb") as fd:
            pattern_counts = pickle.load(fd)
        with open(tree_path, "rb") as fd:
            tree = pickle.load(fd)
    else:
        print("Starting to build predicate sets")
        start = time.time()
        freq_patterns, pattern_counts, tree = cfp_growth(features, predicates_satisfied, min_supports,
                                                         max_depth, max_sets)
        length = time.time() - start
        print(f"{len(freq_patterns)} predicate sets built in {length} seconds")
        print(tree)
        quit()

        print("Starting find closed sets")
        closed_sets = find_closed_predicate_sets(freq_patterns, pattern_counts, predicate_counts,
                                                 min(min_supports.values()), n_workers)
        for i in range(len(closed_sets)):
            print(f"{i + 1}: {len(closed_sets[i])}")

        with open(sets_path, "wb") as fd:
            pickle.dump(closed_sets, fd)
        with open(counts_path, "wb") as fd:
            pickle.dump(pattern_counts, fd)
        with open(tree_path, "wb") as fd:
            pickle.dump(tree, fd)

    rules = generate_rules(closed_sets, pattern_counts, tree)
    print("Number of rules:", len(rules))

    invariants = create_invariant_objs(rules, index_to_predicate)
    return invariants


def generate_rules(closed_sets, predicate_counts, tree, min_confidence=1):
    rules = []
    for i in range(1, len(closed_sets)):
        print(f"Generating rules for {i} / {len(closed_sets)} closed sets")
        for j, freq_sets in enumerate(closed_sets[i]):
            print(f"Running through {j} / {len(closed_sets[i])} frequent patterns")
            predicates = [frozenset([item]) for item in freq_sets]
            if i == 1:
                compute_confidence(freq_sets, predicates, tree, predicate_counts, rules, min_confidence)
            else:
                create_rules(freq_sets, predicates, tree, predicate_counts, rules, min_confidence)
    return rules


def find_closed_predicate_sets(predicate_sets: List[frozenset], set_counts: dict, predicate_counts: dict,
                               min_support: dict, n_workers: int):
    pattern_sizes = defaultdict(list)
    for item in predicate_counts:
        if predicate_counts[item] >= min_support:
            key = frozenset([item])
            set_counts[key] = predicate_counts[item]
            pattern_sizes[0].append(key)

    for pattern in predicate_sets:
        pattern_sizes[len(pattern) - 1].append(frozenset(pattern))

    for i in pattern_sizes:
        print(f"{i + 1}: {len(pattern_sizes[i])}")

    closed_sets = []
    if n_workers > 1:
        tasks, results = mp.JoinableQueue(), mp.JoinableQueue()
        work_events = [mp.Event() for _ in range(n_workers)]
        n_tasks = 0
        for i in range(len(pattern_sizes) - 1):
            closed_sets.append([])
            prev_sets = pattern_sizes[i]
            for prev_set in prev_sets:
                tasks.put((prev_set, i))
                n_tasks += 1

        workers = [
            mp.Process(target=_closed_sets_worker, args=(i, pattern_sizes, tasks, results, work_events, set_counts))
            for i in range(n_workers)]
        for worker in workers:
            worker.start()

        done = False
        counter = 0
        while results.qsize() > 0 or not done:
            try:
                set_, i, is_closed = results.get(timeout=0.1)
                if is_closed:
                    closed_sets[i].append(set_)
                results.task_done()
                counter += 1
                print(end="\r")
                print(f"Completed {counter} / {n_tasks} tasks", flush=True, end="")

                done = True
                for e in work_events:
                    if not e.is_set():
                        done = False
            except queue.Empty:
                pass
        tasks.join()
        results.join()

        for worker in workers:
            worker.join()
    else:
        print("Pattern sizes", len(pattern_sizes))
        for i in range(len(pattern_sizes) - 1):
            start = time.time()
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
            length = time.time() - start
            print(f"Closed sets {i} took {length} seconds with {len(prev_sets)}, {len(next_sets)} sets")
    closed_sets.append(pattern_sizes[len(pattern_sizes) - 1])
    print()
    return closed_sets


def _closed_sets_worker(rank: int, pattern_sizes: dict, tasks: mp.Queue, results: mp.Queue,
                        work_completed_events: List[mp.Event], set_counts: dict):
    while tasks.qsize() > 0:
        try:
            prev_set, i = tasks.get(timeout=0.1)
            is_closed = True
            for next_set in pattern_sizes[i + 1]:
                if prev_set.issubset(next_set) and set_counts[prev_set] == set_counts[next_set]:
                    is_closed = False
                    break
            results.put((prev_set, i, is_closed))
            tasks.task_done()
        except queue.Empty:
            pass
    work_completed_events[rank].set()


def compute_confidence(freq_set, predicates: list, tree: MISTree, pattern_counts: dict, rules: list,
                       min_confidence: float):
    rule_sets = []
    for consequence in predicates:
        antecedent = freq_set - consequence
        print(antecedent, freq_set)
        if antecedent not in pattern_counts:
            pattern_counts[antecedent] = tree.support(list(antecedent))
            print(f"Support: {pattern_counts[antecedent]}")

        confidence = pattern_counts[freq_set] / pattern_counts[antecedent]
        print("confidence:", pattern_counts[freq_set], pattern_counts[antecedent])
        if confidence >= min_confidence:
            rules.append((antecedent, consequence, confidence))
            rule_sets.append(consequence)
    print("rule sets", len(rule_sets))
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


def assign_predicates(predicates: List[Predicate], dataset: ICSDataset, n_workers=1) -> Tuple[dict, dict]:
    """Determine which states satisfy each predicate"""
    features, labels = dataset.get_data()
    predicate_counts = {}
    for p in range(len(predicates)):
        predicate_counts[p] = 0
    predicates_satisfied = {}
    if n_workers > 1:
        task_queue, result_queue = mp.JoinableQueue(), mp.JoinableQueue()
        events = [mp.Event() for _ in range(n_workers)]
        for state in features:
            task_queue.put(state)
        workers = [mp.Process(target=_assign_predicates, args=(i, predicates, task_queue, result_queue, events)) for i
                   in range(n_workers)]
        for worker in workers:
            worker.start()

        done = False
        counter = 0
        while result_queue.qsize() > 0 or not done:
            try:
                state, predicates_result = result_queue.get(timeout=1)
                predicates_satisfied[tuple(state)] = predicates_result
                for i in predicates_result:
                    predicate_counts[i] += 1
                result_queue.task_done()
                counter += 1
                print(end="\r")
                print(f"Assigned {counter} / {len(features)} states", flush=True, end="")

                done = True
                for e in events:
                    if not e.is_set():
                        done = False
            except queue.Empty:
                pass
        task_queue.join()
        result_queue.join()
        for worker in workers:
            worker.join()
    else:
        for i, state in enumerate(features):
            predicates_result = count_predicate_satisfied(predicates, torch.as_tensor(state))
            predicates_satisfied[tuple(state)] = predicates_result
            for i in predicates_result:
                predicate_counts[i] += 1
            print(f"Assigned {i:5d} / {len(features)} states", end="\r", flush=True)
    print()
    return dict(predicate_counts), predicates_satisfied


def _assign_predicates(rank: int, predicates: List[Predicate], tasks: mp.JoinableQueue, results: mp.JoinableQueue,
                       work_completed_events: List[mp.Event]):
    while tasks.qsize() > 0:
        try:
            state = tasks.get(timeout=0.1)
            predicates_satisfied = count_predicate_satisfied(predicates, state)
            results.put((state, predicates_satisfied))
            tasks.task_done()
        except queue.Empty:
            pass
    results.close()
    work_completed_events[rank].set()


def count_predicate_satisfied(predicates: List[Predicate], state: torch.Tensor):
    predicates_satisfied = []
    for i, predicate in enumerate(predicates):
        if type(state) is not torch.Tensor:
            state = torch.as_tensor(state)
        if predicate.is_satisfied(state):
            predicates_satisfied.append(i)
    return predicates_satisfied
