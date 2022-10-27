import pickle
from os import path
import torch
from typing import List
import queue
import multiprocessing as mp

from datasets import ICSDataset, SWATDataset
from models import prediction_loss
from .evaluator import Evaluator
from models import get_losses
from invariants import Invariant


class NNEvaluator(Evaluator):
    def __init__(self, conf: dict, dataset: ICSDataset):
        super(NNEvaluator, self).__init__(conf, dataset)
        print(f"Number of samples: {len(dataset)}")

        self.model_path = path.join(self.checkpoint, "model.pt")
        self.normal_behavior_path = path.join(self.checkpoint, "normal_behavior.pt")
        self.invariants_path = path.join(conf["train"]["invariant_path"] + "_invariants.pkl")
        n_workers = conf["train"]['n_workers']

        info = torch.load(self.model_path)
        self.model = info["model"]
        self.model.eval()
        obj = torch.load(self.normal_behavior_path)
        self.normal_means = obj["mean"]
        self.normal_stds = obj["std"]

        self.loss = conf["train"]["loss"]
        self.invariants = None
        self.workers = None
        self.tasks = None
        self.results = None
        self.work_completed_events = None
        self.input_queue = None
        self.start_work_event = None
        self.end_workers_events = None

        if self.loss == "invariant":
            with open(self.invariants_path, "rb") as fd:
                self.invariants = pickle.load(fd)

            if n_workers > 1:
                self.work_completed_events = [mp.Event() for _ in range(n_workers)]
                self.tasks = mp.JoinableQueue()
                self.results = mp.JoinableQueue()
                self.input_queue = mp.JoinableQueue()
                self.start_work_event = mp.Event()
                self.end_workers_events = mp.Event()
                self.start_work_event.clear()
                self.end_workers_events.clear()

                self.workers = [mp.Process(target=evaluate_invariants, args=(i, self.invariants, self.input_queue, self.tasks,
                                                                             self.results, self.work_completed_events,
                                                                             self.start_work_event, self.end_workers_events))
                                for i in range(n_workers)]
                for worker in self.workers:
                    worker.start()
        self.hidden_states = [None for _ in range(len(self.conf["model"]["hidden_layers"]) - 1)]
        if n_workers > 1:
            self.loss_fns = get_losses(dataset, None, n_workers)
        else:
            self.loss_fns = get_losses(dataset, self.invariants, n_workers)

    def close(self):
        if self.workers is not None:
            self.start_work_event.set()
            self.end_workers_events.set()
            self.input_queue.join()
            self.tasks.join()
            self.results.join()
            for worker in self.workers:
                worker.join()

    def alert(self, state, target):
        state = state.unsqueeze(0)
        losses = self.compute_loss(state, target)
        losses = losses.detach()
        score = torch.abs(losses - self.normal_means) / self.normal_stds

        # scores.append(score.item())
        # print(loss, normal_means, normal_stds, attack)
        alarm = torch.any(score > 2)
        return alarm

    def compute_loss(self, seq, target):
        with torch.no_grad():
            outputs, self.hidden_states = self.model(seq, self.hidden_states)

        loss = []
        for loss_fn in self.loss_fns:
            losses = loss_fn(seq, outputs, target, self.dataset.get_categorical_features())
            loss.append(losses.view(1, -1))

        # Invariants
        if len(self.workers) > 0:
            for _ in self.workers:
                self.input_queue.put((seq, outputs))
            losses = torch.zeros((len(self.invariants)))

            for i in range(len(self.invariants)):
                self.tasks.put(i)

            for e in self.work_completed_events:
                e.clear()
            self.start_work_event.set()

            done = False
            counter = 0
            self.start_work_event.clear()
            while self.results.qsize() > 0 or not done:
                try:
                    i, inv_loss = self.results.get(timeout=0.1)
                    losses[i] = inv_loss
                    # print(f"{counter + 1} / {len(self.invariants)} invariants evaluated", flush=True, end=" ")
                    # print("\r", end="")
                    counter += 1
                    self.results.task_done()
                except queue.Empty:
                    pass

                done = True
                for e in self.work_completed_events:
                    if not e.is_set():
                        done = False
            loss.append(losses.view(1, -1))
        return torch.concat(loss, dim=1)


def evaluate_invariants(rank: int, invariants: List[Invariant], inputs: mp.JoinableQueue, tasks: mp.JoinableQueue,
                        results: mp.JoinableQueue, work_completed_events: List[mp.Event], start_work_event: mp.Event,
                        end_work_event: mp.Event):
    while not end_work_event.is_set():
        start_work_event.wait()
        try:
            batch, outputs = inputs.get(timeout=0.1)
            while tasks.qsize() > 0:
                try:
                    i = tasks.get(timeout=0.1)
                    invariant = invariants[i]
                    loss = invariant.confidence(batch, outputs)
                    results.put((i, loss.item()))
                    tasks.task_done()
                except queue.Empty:
                    pass
            work_completed_events[rank].set()
            inputs.task_done()
        except queue.Empty:
            pass
