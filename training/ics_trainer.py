from os import path
import pickle
import queue

import numpy as np
from pytorch_lightning import LightningModule
import torch
from torch import optim
import multiprocessing as mp
from typing import List
from invariants import Invariant


from models import PredictionModel, get_losses


class ICSTrainer(LightningModule):

    def __init__(self, conf, categorical_values):
        super(ICSTrainer, self).__init__()
        self.save_hyperparameters()
        self.conf = conf
        self.categorical_values = categorical_values
        self.model = PredictionModel(conf, categorical_values)
        self.learning_rate = conf["train"]["lr"]
        self.decay = conf["train"]["regularization"]
        self.loss = conf["train"]["loss"]
        self.n_workers = conf["train"]['n_workers']
        self.checkpoint_dir = path.join("checkpoint", conf["train"]["checkpoint"])
        self.normal_behavior_path = path.join(self.checkpoint_dir, "normal_behavior.pt")
        self.invariants_path = path.join("checkpoint", conf["train"]["invariants"] + "_invariants.pkl")

        self.hidden_states = None
        self.invariants = None

        if self.loss == "invariant":
            with open(self.invariants_path, "rb") as fd:
                self.invariants = pickle.load(fd)
        self.loss_fns = get_losses(self.invariants)
        self.normal_losses = []

    def forward(self, x, hidden_states):
        return self.model(x, hidden_states)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.decay)
        return optimizer

    def training_step(self, batch, batch_idx):
        self.hidden_states = [None for _ in range(len(self.conf["model"]["hidden_layers"]) - 1)]
        losses = self.compute_loss(*batch)
        loss = torch.sum(losses, dim=1)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        self.hidden_states = [None for _ in range(len(self.conf["model"]["hidden_layers"]) - 1)]
        losses = self.compute_loss(*batch)
        loss = torch.sum(losses, dim=1)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        losses = self.compute_loss(*batch)
        self.normal_losses.append(losses)

    def on_test_start(self):
        self.hidden_states = [None for _ in range(len(self.conf["model"]["hidden_layers"]) - 1)]

    def on_test_end(self):
        losses = torch.concat(self.normal_losses, dim=0)
        means = torch.mean(losses, dim=0).numpy().flatten()
        stds = torch.std(losses, dim=0).numpy().flatten()

        obj = {
            "mean": means,
            "std": stds
        }

        torch.save(obj, self.normal_behavior_path)

    def compute_loss(self, seq, target):
        outputs, self.hidden_states = self.model(seq, self.hidden_states)
        loss = []
        for loss_fn in self.loss_fns:
            losses = loss_fn(seq, outputs, target, self.categorical_values)
            loss.append(losses.view(1, -1))
        return torch.concat(loss, dim=1)
