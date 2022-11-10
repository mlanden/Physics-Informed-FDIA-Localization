import multiprocessing as mp
import pickle
import queue
from os import path
from typing import List
import json
import joblib
import torch

from datasets import ICSDataset
from invariants import Invariant
from models import prediction_loss, invariant_loss
from .evaluator import Evaluator


class NNEvaluator(Evaluator):
    def __init__(self, conf: dict, model: torch.nn.Module,  dataset: ICSDataset):
        super(NNEvaluator, self).__init__(conf, dataset)
        print(f"Number of samples: {len(dataset)}")
        self.gmm = True

        self.normal_model_path = path.join(self.checkpoint_dir, "normal_model.gz")
        self.losses_path = path.join(self.checkpoint_dir, "evaluation_losses.json")
        self.invariants_path = path.join("checkpoint", conf["train"]["invariants"] + "_invariants.pkl")
        self.n_workers = conf["train"]['n_workers']
        self.model_path = path.join(self.checkpoint_dir, "model.pt")
        self.loss = conf["train"]["loss"]
        self.normal_behavior_path = path.join(self.checkpoint_dir, "normal_behavior.json")
        if self.gmm:
            self.normal_model = joblib.load(self.normal_model_path)
            with open(self.normal_behavior_path, "r") as fd:
                self.min_score = json.load(fd)[0]
        else:
            with open(self.normal_behavior_path, "r") as fd:
                obj = json.load(fd)
            self.normal_means = torch.as_tensor(obj["mean"])
            self.normal_stds = torch.as_tensor(obj["std"])

        self.model = model
        self.model.eval()
        self.saved_losses = []
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

            if self.n_workers > 1:
                self.work_completed_events = [mp.Event() for _ in range(self.n_workers)]
                self.tasks = mp.JoinableQueue()
                self.results = mp.JoinableQueue()
                self.input_queue = mp.JoinableQueue()
                self.start_work_event = mp.Event()
                self.end_workers_events = mp.Event()
                self.start_work_event.clear()
                self.end_workers_events.clear()

                self.workers = [mp.Process(target=evaluate_invariants,
                                           args=(i, self.invariants, self.input_queue, self.tasks, self.results,
                                                 self.work_completed_events, self.start_work_event,
                                                 self.end_workers_events))
                                for i in range(self.n_workers)]
                for worker in self.workers:
                    worker.start()
        self.hidden_states = [None for _ in range(len(self.conf["model"]["hidden_layers"]) - 1)]
        self.loss_fns = [prediction_loss]
        if self.invariants is not None and self.n_workers > 0:
            self.loss_fns.append(invariant_loss)

    def close(self):
        if self.workers is not None:
            self.start_work_event.set()
            self.end_workers_events.set()
            self.input_queue.join()
            self.tasks.join()
            self.results.join()
            for worker in self.workers:
                worker.join()

    def alert(self, state, target, attack):
        unscaled_state, scaled_state = state
        unscaled_state = unscaled_state.unsqueeze(0)
        scaled_state = scaled_state.unsqueeze(0)
        losses = self.compute_loss(unscaled_state, scaled_state, target)
        losses = losses.detach()

        if self.gmm:
            score = self.normal_model.score_samples(losses.numpy())
            alarm = score < 0.5 * self.min_score
            if attack and not alarm:
                print(score / self.min_score)
            self.saved_losses.append((score[0], attack.float().item()))
        else:
            score = torch.abs(losses - self.normal_means) / self.normal_stds
            alarm = torch.any(score > 2)

        # print(" ", attack, score, self.min_score)
        return alarm

    def on_evaluate_end(self):
        with open(self.losses_path, "w") as fd:
            json.dump(self.saved_losses, fd)

    def compute_loss(self, unscaled_state, scaled_state, target):
        with torch.no_grad():
            outputs, self.hidden_states = self.model((unscaled_state, scaled_state), self.hidden_states)

        loss = []
        for loss_fn in self.loss_fns:
            losses = loss_fn(unscaled_state, outputs, target, self.dataset.get_categorical_features())
            loss.append(losses.view(1, -1))

        # Invariants
        if self.workers is not None:
            for _ in self.workers:
                self.input_queue.put((unscaled_state, outputs))
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
            unscaled_states, outputs = inputs.get(timeout=0.1)
            while tasks.qsize() > 0:
                try:
                    i = tasks.get(timeout=0.1)
                    invariant = invariants[i]
                    loss = invariant.confidence(unscaled_states, outputs)
                    results.put((i, loss.item()))
                    tasks.task_done()
                except queue.Empty:
                    pass
            work_completed_events[rank].set()
            inputs.task_done()
        except queue.Empty:
            pass
