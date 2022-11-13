import multiprocessing as mp
import pickle
import queue
from functools import partial
from os import path
from typing import List
import json
import joblib
import numpy as np
import torch

from datasets import ICSDataset
from invariants import Invariant
from models import prediction_loss, invariant_loss
from .evaluator import Evaluator
from utils import make_roc_curve
from invariants import evaluate_invariants

eps = torch.finfo(torch.float32).eps


class NNEvaluator(Evaluator):
    def __init__(self, conf: dict, model: torch.nn.Module,  dataset: ICSDataset):
        super(NNEvaluator, self).__init__(conf, dataset)
        print(f"Number of samples: {len(dataset)}")

        self.normal_model_path = path.join(self.checkpoint_dir, "normal_model.gz")
        self.losses_path = path.join(self.checkpoint_dir, "evaluation_losses.json")#_invariant_std.json")
        self.invariants_path = path.join("checkpoint", conf["train"]["invariants"] + "_invariants.pkl")
        self.n_workers = conf["train"]['n_workers']
        self.model_path = path.join(self.checkpoint_dir, "model.pt")
        self.loss = conf["train"]["loss"]
        self.profile_type = conf["model"]["profile_type"]
        self.normal_behavior_path = path.join(self.checkpoint_dir, "normal_behavior.json")
        print("Loading normal profile from", self.normal_behavior_path)
        if self.profile_type == "gmm":
            self.normal_model = joblib.load(self.normal_model_path)
            with open(self.normal_behavior_path, "r") as fd:
                self.min_score = json.load(fd)[0]
        elif self.profile_type == "mean":
            with open(self.normal_behavior_path, "r") as fd:
                obj = json.load(fd)
            self.normal_means = torch.as_tensor(obj["mean"])
            self.normal_stds = torch.as_tensor(obj["std"])
        else:
            raise RuntimeError("Unknown profile type")

        self.model = model
        self.model.eval()
        self.saved_losses = []
        self.invariants = None

        if self.loss == "invariant":
            print("Invariant setup")
            with open(self.invariants_path, "rb") as fd:
                self.invariants = pickle.load(fd)
                print(f"Loaded {len(self.invariants)} invariants")

        self.hidden_states = [None for _ in range(len(self.conf["model"]["hidden_layers"]) - 1)]
        self.loss_fns = [prediction_loss]
        if self.invariants is not None and self.n_workers == 0:
            self.loss_fns.append(partial(invariant_loss, invariants=self.invariants))

    def process_state(self, state, target):
        unscaled_state, scaled_state = state

        with torch.no_grad():
            outputs, self.hidden_states = self.model((unscaled_state, scaled_state), self.hidden_states)

        loss = []
        for loss_fn in self.loss_fns:
            losses = loss_fn(unscaled_state, outputs, target, self.dataset.get_categorical_features())
            loss.append(losses.view(1, -1))

        return outputs, loss

    def alert(self, states, targets, attacks, intermediates):
        outputs, losses = zip(*intermediates)
        losses = torch.concat([loss[0] for loss in losses], dim=0)
        print(losses.shape)

        if self.invariants is not None:
            loss = evaluate_invariants(self.invariants, states, outputs, self.n_workers)
            losses = torch.concat([losses, loss], dim=1)

        alarm = False
        if self.profile_type == "gmm":
            scores = self.normal_model.score_samples(losses.numpy())
            alarm = scores < 0.5 * self.min_score
            # if attack and not alarm:
            #     print(score / self.min_score)
            for score, attack in zip(scores, attacks):
                self.saved_losses.append((score[0] / self.min_score, attack.float().item()))
        elif self.profile_type == "mean":
            scores = torch.abs(losses - self.normal_means) / (self.normal_stds + eps)
            alarm = torch.any(scores > 2.1, dim=1)
            # if not attack and alarm:
            #     print(torch.max(score))
            for score, attack in zip(scores, attacks):
                self.saved_losses.append((torch.max(score).item(), attack.float().item()))

        # print(" ", attack, score, self.min_score)
        return alarm

    def on_evaluate_end(self):
        with open(self.losses_path, "w") as fd:
            json.dump(self.saved_losses, fd)

        make_roc_curve(self.losses_path)

