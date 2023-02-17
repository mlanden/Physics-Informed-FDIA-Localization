import json
import os
import sys

import time
import traceback
from typing import List, Tuple
from collections import defaultdict
import torch.multiprocessing as mp

import numpy as np
import pickle
from os import path
from typing import List
import queue

import torch
from .invariant import Invariant
from datasets import ICSDataset
from utils import cfp_growth
from .distribution_predicate import DistributionPredicate
from sklearn.metrics import confusion_matrix

sys.setrecursionlimit(50000)


class InvariantMiner:
    def __init__(self, conf: dict, dataset: ICSDataset):
        self.dataset = dataset
        self.conf = conf
        self.checkpoint = conf["train"]["checkpoint"]
        self.load_checkpoint = conf["train"]["load_checkpoint"]
        self.predicate_path = path.join("checkpoint", self.checkpoint, "predicates.pkl")
        self.n_workers = conf["train"]["n_workers"]
        self.invariant_fraction = conf["train"]["invariant_fraction"]
        self.validate_fraction = conf["train"]["validate_fraction"]
        self.local_min_support = conf["train"]["gamma"]
        self.global_min_support = conf["train"]["theta"]
        self.max_depth = conf["train"]["max_depth"]
        self.min_confidence = conf["train"]["min_confidence"]

        if not path.exists(self.predicate_path):
            raise RuntimeError("No predicates found, create them first")

        with open(self.predicate_path, "rb") as fd:
            self.predicates = pickle.load(fd)
        print(f"Loaded {len(self.predicates)} predicates")

        self.assign_path = path.join("checkpoint", self.checkpoint, "assigned_predicates.pkl")
        self.count_path = path.join("checkpoint", self.checkpoint, "count_predicates.pkl")
        self.sets_path = path.join("checkpoint", self.checkpoint, "predicate_sets.pkl")
        self.counts_path = path.join("checkpoint", self.checkpoint, "predicate_counts.pkl")
        self.tree_path = path.join("checkpoint", self.checkpoint, "tree.pkl")
        self.invariants_path = path.join("checkpoint", conf["train"]["invariants"] + "_invariants.pkl")
        self.result_path = path.join("results", self.checkpoint, "invariants")

        self.dataset_size = 0
        self.invalid_invariants = 0
        self.features = None
        self.predicate_counts = None
        self.predicates_satisfied = None
        self.tree = None
        self.invariants = None

    def mine_invariants(self):
        self.invariants = []
        predicates = self.predicates

        # Mode 1
        self.predicates = []
        for p in predicates:
            if type(p) is DistributionPredicate:
                self.predicates.append(p)
        self._add_rules()

        # Mode 2
        self.predicates = []
        for p in predicates:
            if type(p) is not DistributionPredicate:
                self.predicates.append(p)
        self._add_rules()
        print(f"Invalid invariants: {self.invalid_invariants}")
        # self.evaluate(score=False)

        print(f"Saved {len(self.invariants)} invariants")
        with open(self.invariants_path, "wb") as fd:
            pickle.dump(self.invariants, fd)

    def _add_rules(self):
        self.features, labels = self.dataset.get_data()
        print(self.features.shape)
        self.dataset_size = int(len(self.features) * self.invariant_fraction)
        # self.features = self.features[:self.dataset_size, :]
        print(f"Using {len(self.predicates)} predicates")
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

        print("Predicates assigned")

        min_supports = {}
        for i, predicate in enumerate(self.predicates):
            if self.predicate_counts[i] > 0:
                min_supports[i] = max(self.local_min_support * self.predicate_counts[i], len(self.features) *
                                      self.global_min_support)
                # print(min_supports[i])
            else:
                min_supports[i] = len(self.features)

        for i in self.predicates_satisfied.values():
            for pred in i:
                assert pred < len(self.predicates), f"{pred} not a predicate"
        self.index_to_predicate, self.predicate_to_index = self.create_mappings()
        print("Mean min support", np.mean(list(min_supports.values())), "Number of states:", len(self.features))
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
        print("Number of rules:", len(self.invariants))

    def generate_rules(self, closed_sets, predicate_counts):
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

            valid = True
            for p in antecedent:
                if p in consequence:
                    valid = False
                    break
            confidence = pattern_counts[freq_set] / pattern_counts[antecedent]
            if confidence >= self.min_confidence:
                if valid:
                    rule = Invariant(antecedent, consequence, pattern_counts[freq_set] / self.dataset_size, self.index_to_predicate)
                    self.invariants.append(rule)
                    rule_sets.append(consequence)
                else:
                    self.invalid_invariants += 1
        return rule_sets

    def create_rules(self, freq_set: frozenset, predicates: List, pattern_counts: dict):
        k = len(predicates[0])
        if len(freq_set) > k + 1:
            joined_sets = create_apriori_sets(predicates, k)
            rule_sets = self.compute_confidence(freq_set, joined_sets, pattern_counts)
            if len(rule_sets) > 1:
                self.create_rules(freq_set, rule_sets, pattern_counts)

    def create_mappings(self):
        index_to_predicate = {}
        predicate_to_index = {}
        for i in range(len(self.predicates)):
            index_to_predicate[i] = self.predicates[i]
            predicate_to_index[self.predicates[i]] = i
        return index_to_predicate, predicate_to_index

    def assign_predicates(self) -> Tuple[dict, dict]:
        """Determine which states satisfy each predicate"""
        predicate_counts = {}
        for p in range(len(self.predicates)):
            predicate_counts[p] = 0
        predicates_satisfied = defaultdict(list)

        if self.n_workers > 1:
            task_queue, result_queue = mp.JoinableQueue(), mp.JoinableQueue()
            events = [mp.Event() for _ in range(self.n_workers)]
            for p in range(len(self.predicates)):
                task_queue.put(p)
            workers = [mp.Process(target=self._assign_predicates, args=(i, self.features, task_queue, result_queue, events))
                       for i in range(self.n_workers)]
            for worker in workers:
                worker.start()

            done = False
            counter = 0
            while result_queue.qsize() > 0 or not done:
                try:
                    p_idx, predicates_result = result_queue.get(timeout=1)
                    for state_idx in range(len(self.features)):
                        if predicates_result[state_idx]:
                            predicate_counts[p_idx] += 1
                            predicates_satisfied[state_idx].append(p_idx)

                    result_queue.task_done()
                    counter += 1
                    print(end="\r")
                    print(f"Assigned {counter} / {len(self.predicates)} predicates", flush=True, end="")

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
            for i, p in enumerate(self.predicates):
                res = p.is_satisfied(self.features)
                for state_idx in range(len(self.features)):
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

    def evaluate(self, score=True):
        self.features, labels = self.dataset.get_data()
        if not score:
            # Use validation states
            val_len = int(self.validate_fraction * len(self.features))
            self.features = self.features[self.dataset_size: self.dataset_size + val_len, :]
            labels = labels.iloc[self.dataset_size: self.dataset_size + val_len]
        self.assign_path = path.join("results", self.checkpoint, "assigned_predicates.pkl")
        if self.load_checkpoint and path.exists(self.assign_path):
            with open(self.assign_path, "rb") as fd:
                assignments = pickle.load(fd)
        else:
            self.predicate_counts, self.predicates_satisfied = self.assign_predicates()
            assignments = np.zeros((len(self.features), len(self.predicates) + 2))
            for state in self.predicates_satisfied:
                for p in self.predicates_satisfied[state]:
                    assignments[state, p] = 1

            with open(self.assign_path, "wb") as fd:
                pickle.dump(assignments, fd)

        if self.invariants is not None:
            invariants = self.invariants
        else:
            with open(self.invariants_path, "rb") as fd:
                invariants = pickle.load(fd)

        print(f"There are {len(self.features)} states.")
        alerts = np.zeros((len(self.features)))
        false_positives = [0 for _ in invariants]
        for i, invariant in enumerate(invariants):
            assignments[:, -2] = 1
            assignments[:, -1] = 1
            for predicate in invariant.antecedent:
                assignments[assignments[:, predicate] == 0, -2] = 0

            for predicate in invariant.consequent:
                assignments[assignments[:, predicate] == 0, -1] = 0

            alerts[(assignments[:, -2] == 1) & (assignments[:, -1] == 0)] = 1
            false_positives[i] = np.count_nonzero((assignments[:, -2] == 1) & (assignments[:, -1] == 0) & ~labels)
            print(f"\rCompleted {i} / {len(invariants)} invariants", end="")

        print()
        if score:
            cm = confusion_matrix(labels, alerts)
            tp = cm[1][1]
            fp = cm[0][1]
            fn = cm[1][0]
            tn = cm[0][0]

            tpr = tp / (tp + fn)
            tnr = tn / (tn + fp)
            fpr = fp / (fp + tn)
            fnr = fn / (fn + tp)
            recall = tpr
            precision = tp / (tp + fp)
            f1 = 2 * precision * recall / (precision + recall)
            accuracy = (tp + tn) / len(alerts)
            print(f"True Positive: {tpr * 100 :3.2f}")
            print(f"True Negative: {tnr * 100 :3.2f}")
            print(f"False Positive: {fpr * 100 :3.2f}")
            print(f"False Negatives: {fnr * 100 :3.2f}")
            print(f"F1 Score: {f1 * 100 :3.2f}")
            print(f"Precision: {precision * 100 :3.2f}")
            print(f"Accuracy: {accuracy * 100 :3.2f}")
        else:
            # Remove invariants above 75% quartile
            threshold = np.quantile(false_positives, 0.1)
            for i in reversed(range(len(self.invariants))):
                if false_positives[i] > threshold:
                    del self.invariants[i]

            with open("fp_invariants.json", "w") as fd:
                json.dump(false_positives, fd)


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


def evaluate_invariants(invariants: List[Invariant], states: torch.Tensor, outputs: List[List[torch.Tensor]],
                        n_workers) -> torch.Tensor:
    if n_workers <= 0:
        raise RuntimeError("Cannot use multiprocessing for 0 worker")

    tasks = mp.JoinableQueue()
    results = mp.JoinableQueue()
    work_completed_events = [mp.Event() for _ in range(n_workers)]
    stop_event = mp.Event()
    for i in range(len(invariants)):
        tasks.put(i)
    print(f"Number of tasks: {tasks.qsize()}", flush=True)
    n_tasks = tasks.qsize()

    workers = [mp.Process(target=_invariant_worker, args=(i, invariants, states, outputs, tasks, results,
                                                          work_completed_events, stop_event)) for i in range(n_workers)]
    for worker in workers:
        worker.start()

    count = 0
    losses = torch.zeros((len(invariants), states.shape[0]))
    while count < n_tasks:
        try:
            id_, confidence = results.get(timeout=0.1)
            losses[id_, :] = confidence
            results.task_done()
            count += 1
            print("\r", end="", flush=True)
            print(f"{count} / {len(invariants)} invariants completed", end="", flush=True)
        except queue.Empty:
            pass
        except Exception as e:
            print("Main", e)
    stop_event.set()
    tasks.join()
    results.join()
    for worker in workers:
        worker.join()
    losses = torch.t(losses)
    print("\nInvariants evaluated")
    return losses


def _invariant_worker(rank: int, invariants: List[Invariant], states: torch.Tensor, outputs: List[torch.Tensor],
                      tasks: mp.JoinableQueue, results: mp.JoinableQueue, worker_end_events: List[mp.Event],
                      stop_event: mp.Event):
    while tasks.qsize() > 0:
        try:
            inv_id = tasks.get(timeout=0.1)
            invariant = invariants[inv_id]
            confidence = invariant.confidence(states, outputs)
            results.put((inv_id, confidence))
            tasks.task_done()
        except queue.Empty:
            pass
        except Exception as e:
            traceback.print_exc()

    worker_end_events[rank].set()
    stop_event.wait()
