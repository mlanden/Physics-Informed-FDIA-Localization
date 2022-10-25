import os
import sys

import time
import queue
from typing import List, Tuple
from collections import defaultdict
import multiprocessing as mp

import numpy as np
import pickle
from os import path
import matplotlib

from .invariant import Invariant
from datasets import ICSDataset
from utils import cfp_growth, save_results

matplotlib.use("agg")
import matplotlib.pyplot as plt

sys.setrecursionlimit(50000)


class InvariantMiner:
    def __init__(self, conf: dict, dataset: ICSDataset):
        self.dataset = dataset
        self.conf = conf
        self.checkpoint = conf["train"]["checkpoint"]
        self.load_checkpoint = conf["train"]["load_checkpoint"]
        self.predicate_path = path.join("checkpoint", self.checkpoint, "predicates.pkl")
        self.n_workers = conf["train"]["n_workers"]
        self.local_min_support = conf["train"]["gamma"]
        self.global_min_support = conf["train"]["theta"]
        self.max_depth = conf["train"]["max_depth"]
        self.min_confidence = conf["train"]["min_confidence"]

        if not path.exists(self.predicate_path):
            raise RuntimeError("No predicates found, create them first")

        with open(self.predicate_path, "rb") as fd:
            self.predicates = pickle.load(fd)
        print(f"Loaded {len(self.predicates)} predicates")

        self.index_to_predicate, self.predicate_to_index = self.create_mappings()
        self.assign_path = path.join(self.checkpoint, "assigned_predicates.pkl")
        self.count_path = path.join(self.checkpoint, "count_predicates.pkl")
        self.sets_path = path.join(self.checkpoint, "predicate_sets.pkl")
        self.counts_path = path.join(self.checkpoint, "predicate_counts.pkl")
        self.tree_path = path.join(self.checkpoint, "tree.pkl")
        self.invariants_path = path.join(conf["train"]["invariant_path"] + "_invariants.pkl")
        self.result_path = path.join("results", self.checkpoint, "invariants")

        self.predicate_counts = None
        self.predicates_satisfied = None
        self.tree = None
        self.rules = None

    def mine_invariants(self):
        if path.exists(self.assign_path) and self.load_checkpoint:
            with open(self.assign_path, "rb") as fd:
                self.predicates_satisfied = pickle.load(fd)
            with open(self.count_path, "rb") as fd:
                self.predicate_counts = pickle.load(fd)
        else:
            self.predicate_counts, self.predicates_satisfied = self.assign_predicates()
            with open(self.assign_path, "wb") as fd:
                pickle.dump(self.predicates_satisfied, fd)
            with open(self.count_path, "wb") as fd:
                pickle.dump(self.predicate_counts, fd)
        features, labels = self.dataset.get_data()

        print("Predicates assigned")
        # print(predicates_satisfied.values())
        # data = defaultdict(lambda: 0)
        # for i, state in enumerate(features):
        #     data[i] += len(predicates_satisfied[i])
        #     print(data.values())
        # fig, ax = plt.subplots()
        # ax.boxplot(data.values())
        # # ax.hist(predicate_counts.values(), bins=25)
        # # plt.savefig("predicates.png")
        # plt.savefig("Assigned-predicates.png")
        # quit()

        min_supports = {}
        deletes = []
        for i, predicate in enumerate(self.predicates):
            if self.predicate_counts[i] > 0:
                min_supports[i] = max(self.local_min_support * self.predicate_counts[i], len(features) *
                                      self.global_min_support)
            else:
                deletes.append(i)

        # for i in reversed(deletes):
        #     del self.predicates[i]
        #     del self.predicate_counts[i]

        print("Mean min support", np.mean(list(min_supports.values())), "Number of states:", len(features))
        if path.exists(self.sets_path) and self.load_checkpoint:
            with open(self.sets_path, "rb") as fd:
                closed_sets = pickle.load(fd)
            with open(self.counts_path, "rb") as fd:
                pattern_counts = pickle.load(fd)
            with open(self.tree_path, "rb") as fd:
                self.tree = pickle.load(fd)
        else:
            print("Starting to build predicate sets")
            start = time.time()
            freq_patterns, pattern_counts, self.tree = cfp_growth(self.predicates_satisfied, min_supports, self.max_depth)
            length = time.time() - start
            print(f"{len(freq_patterns)} predicate sets built in {length} seconds")
            with open(self.counts_path, "wb") as fd:
                pickle.dump(pattern_counts, fd)
            with open(self.tree_path, "wb") as fd:
                pickle.dump(self.tree, fd)

            print("Starting find closed sets")
            closed_sets = self.find_closed_predicate_sets(freq_patterns, pattern_counts, min(min_supports.values()))
            for i in range(len(closed_sets)):
                print(f"{i + 1}: {len(closed_sets[i])}")

            with open(self.sets_path, "wb") as fd:
                pickle.dump(closed_sets, fd)

        self.generate_rules(closed_sets, pattern_counts)
        print("Number of rules:", len(self.rules))

        # invariants = self.create_invariant_objs()
        with open(self.invariants_path, "wb") as fd:
            pickle.dump(self.rules, fd)

    def generate_rules(self, closed_sets, predicate_counts):
        self.rules = []
        for i in range(1, len(closed_sets)):
            print(f"Generating rules for {i} / {len(closed_sets)} closed sets")
            for j, freq_sets in enumerate(closed_sets[i]):
                predicates = [frozenset([item]) for item in freq_sets]
                if i == 1:
                    self.compute_confidence(freq_sets, predicates, predicate_counts)
                else:
                    self.create_rules(freq_sets, predicates, predicate_counts)

    def find_closed_predicate_sets(self, predicate_sets: List[frozenset], set_counts: dict,
                                   min_support: dict):
        pattern_sizes = defaultdict(list)
        for item in self.predicate_counts:
            if self.predicate_counts[item] >= min_support:
                key = frozenset([item])
                set_counts[key] = self.predicate_counts[item]
                pattern_sizes[0].append(key)

        for pattern in predicate_sets:
            pattern_sizes[len(pattern) - 1].append(frozenset(pattern))

        for i in pattern_sizes:
            print(f"{i + 1}: {len(pattern_sizes[i])}")

        closed_sets = []
        if self.n_workers > 1:
            tasks, results = mp.JoinableQueue(), mp.JoinableQueue()
            work_events = [mp.Event() for _ in range(self.n_workers)]
            n_tasks = 0
            for i in range(len(pattern_sizes) - 1):
                closed_sets.append([])
                prev_sets = pattern_sizes[i]
                for prev_set in prev_sets:
                    tasks.put((prev_set, i))
                    n_tasks += 1

            if n_tasks > 0:
                workers = [
                    mp.Process(target=_closed_sets_worker,
                               args=(i, pattern_sizes, tasks, results, work_events, set_counts))
                    for i in range(self.n_workers)]
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
                    except queue.Empty:
                        pass

                    done = True
                    for e in work_events:
                        if not e.is_set():
                            done = False

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

    def compute_confidence(self, freq_set, predicates: list, pattern_counts: dict):
        rule_sets = []
        for consequence in predicates:
            if len(consequence) == len(freq_set):
                continue
            antecedent = freq_set - consequence
            if antecedent not in pattern_counts:
                pattern_counts[antecedent] = self.tree.support(list(antecedent))

            confidence = pattern_counts[freq_set] / pattern_counts[antecedent]
            if confidence >= self.min_confidence:
                rule = Invariant(antecedent, consequence, self.index_to_predicate)
                self.rules.append(rule)
                rule_sets.append(consequence)
        return rule_sets

    def create_rules(self, freq_set: frozenset, predicates: List, pattern_counts: dict):
        k = len(predicates[0])
        if len(freq_set) > k + 1:
            joined_sets = create_apriori_sets(predicates, k)
            rule_sets = self.compute_confidence(freq_set, joined_sets, pattern_counts)
            if len(rule_sets) > 1:
                self.create_rules(freq_set, rule_sets, pattern_counts)

    def create_invariant_objs(self):
        invariants = []
        for antecedent, consequent, confidence in self.rules:
            antecedents = frozenset([self.index_to_predicate[i] for i in antecedent])
            consequents = frozenset([self.index_to_predicate[i] for i in consequent])
            invariants.append(Invariant(antecedents, consequents))
        return invariants

    def create_mappings(self):
        index_to_predicate = {}
        predicate_to_index = {}
        for i in range(len(self.predicates)):
            index_to_predicate[i] = self.predicates[i]
            predicate_to_index[self.predicates[i]] = i
        return index_to_predicate, predicate_to_index

    def assign_predicates(self) -> Tuple[dict, dict]:
        """Determine which states satisfy each predicate"""
        features, labels = self.dataset.get_data()
        predicate_counts = {}
        for p in range(len(self.predicates)):
            predicate_counts[p] = 0
        predicates_satisfied = defaultdict(list)

        if self.n_workers > 1:
            task_queue, result_queue = mp.JoinableQueue(), mp.JoinableQueue()
            events = [mp.Event() for _ in range(self.n_workers)]
            for p in range(len(self.predicates)):
                task_queue.put(p)
            workers = [mp.Process(target=self._assign_predicates, args=(i, features, task_queue, result_queue, events))
                       for i in range(self.n_workers)]
            for worker in workers:
                worker.start()

            done = False
            counter = 0
            while result_queue.qsize() > 0 or not done:
                try:
                    p_idx, predicates_result = result_queue.get(timeout=1)
                    for state_idx in range(len(features)):
                        if predicates_result[state_idx]:
                            predicate_counts[p_idx] += 1
                            predicates_satisfied[state_idx].append(p_idx)

                    result_queue.task_done()
                    counter += 1
                    print(end="\r")
                    print(f"Assigned {counter} / {len(self.predicates)} predicates", flush=True, end="")
                except queue.Empty:
                    pass

                done = True
                for e in events:
                    if not e.is_set():
                        done = False
            task_queue.join()
            result_queue.join()
            for worker in workers:
                worker.join()
        else:
            for i, p in enumerate(self.predicates):
                res = p.is_satisfied(features)
                for state_idx in range(len(features)):
                    if res[state_idx]:
                        predicate_counts[i] += 1
                        predicates_satisfied[state_idx].append(i)
                print(f"Assigned {i:5d} / {len(self.predicates)} predicates", end="\r", flush=True)
        print()
        return dict(predicate_counts), predicates_satisfied

    def _assign_predicates(self, rank: int, features: np.ndarray, tasks: mp.JoinableQueue,
                           results: mp.JoinableQueue,
                           work_completed_events: List[mp.Event]):
        while tasks.qsize() > 0:
            try:
                p_idx = tasks.get(timeout=0.1)
                predicates_satisfied = self.predicates[p_idx].is_satisfied(features)
                results.put((p_idx, predicates_satisfied))
                tasks.task_done()
            except queue.Empty:
                pass
        results.close()
        work_completed_events[rank].set()

    def evaluate(self):
        features, labels = self.dataset.get_data()
        self.assign_path = path.join("results", self.checkpoint, "assigned_predicates.pkl")
        if self.load_checkpoint and path.exists(self.assign_path):
            with open(self.assign_path, "rb") as fd:
                assignments = pickle.load(fd)
        else:
            self.predicate_counts, self.predicates_satisfied = self.assign_predicates()
            assignments = np.zeros((len(features), len(self.predicate_counts) + 2))
            for state in self.predicates_satisfied:
                for p in self.predicates_satisfied[state]:
                    assignments[state, p] = 1

            with open(self.assign_path, "wb") as fd:
                pickle.dump(assignments, fd)

        ante, conseq = [], []
        with open(self.invariants_path, "rb") as fd:
            invariants = pickle.load(fd)

        print(f"There are {len(features)} states.")
        alerts = np.zeros((len(features)))
        for i, invariant in enumerate(invariants[:50000]):
            assignments[:, -2] = 1
            assignments[:, -1] = 1
            for predicate in invariant.antecedent:
                assignments[assignments[:, predicate] == 0, -2] = 0

            for predicate in invariant.consequent:
                assignments[assignments[:, predicate] == 0, -1] = 0

            alerts[(assignments[:, -2] == 1) & (assignments[:, -1] == 0)] = 1
            print(f"\rCompleted {i} / {len(invariants)} invariants", end="")

            # print(
            #     f"Complete {inv_idx} / {len(invariants)} invariants. Unmatched {antecedents_unmatched},")  # , end="\r")

        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for alert, attack in zip(alerts, labels):
            if attack:
                if alert:
                    tp += 1
                else:
                    fn += 1
            else:
                if alert:
                    fp += 1
                else:
                    tn += 1
        if not path.exists(self.result_path):
            os.makedirs(self.result_path)
        print()
        save_results(tp, tn, fp, fn, labels.tolist(), self.result_path)


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
