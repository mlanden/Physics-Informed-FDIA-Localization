import pickle
import joblib
import json
from os import path

import numpy as np
from sklearn.mixture import GaussianMixture


import torch
from pytorch_lightning import LightningModule
from torch import optim

from models import PredictionModel, invariant_loss, prediction_loss
from invariants import evaluate_invariants


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
        self.max_epochs = conf["train"]["epochs"]
        self.max_gmm_components = conf["train"]["max_gmm_components"]
        self.checkpoint_dir = path.join("checkpoint", conf["train"]["checkpoint"])
        self.normal_behavior_path = path.join(self.checkpoint_dir, "normal_behavior.json")
        self.normal_model_path = path.join(self.checkpoint_dir, "normal_model.gz")
        self.invariants_path = path.join("checkpoint", conf["train"]["invariants"] + "_invariants.pkl")

        self.hidden_states = None
        self.recent_outputs = None
        self.invariants = None

        self.loss_fns = None
        if self.loss == "invariant":
            with open(self.invariants_path, "rb") as fd:
                self.invariants = pickle.load(fd)

        self.normal_losses = []
        self.states = []
        self.outputs = []

    def forward(self, x, hidden_states):
        return self.model(*x, hidden_states)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)# , weight_decay=self.decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=self.max_epochs,
                                                               eta_min=self.learning_rate / 50)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        self.hidden_states = [None for _ in range(len(self.conf["model"]["hidden_layers"]) - 1)]
        losses = self.compute_loss(*batch)
        loss = torch.sum(losses, dim=1)
        self.log("train_loss", loss)
        return loss

    def on_validation_start(self):
        self.loss_fns = [prediction_loss]
        if self.invariants is not None:
            self.loss_fns.append(invariant_loss)

    def validation_step(self, batch, batch_idx):
        self.hidden_states = [None for _ in range(len(self.conf["model"]["hidden_layers"]) - 1)]
        losses = self.compute_loss(*batch)
        loss = torch.sum(losses, dim=1)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        losses = self.compute_loss(*batch)
        self.states.append(batch[0].cpu().view(1, -1))
        self.outputs.append(self.recent_outputs)
        self.normal_losses.append(losses)
        return self.recent_outputs

    def on_test_start(self):
        self.hidden_states = [None for _ in range(len(self.conf["model"]["hidden_layers"]) - 1)]
        self.loss_fns = [prediction_loss]
        if self.n_workers == 0 and self.invariants is not None:
            self.loss_fns.append(invariant_loss)
        print(f"{len(self.loss_fns)} loss functions")

    def on_test_end(self):
        losses = torch.concat(self.normal_losses, dim=0)
        if self.n_workers > 0 and self.invariants is not None:
            print(f"Evaluating invariants with {self.n_workers} workers", flush=True)
            states = torch.concat(self.states, dim=0)
            invariant_losses = evaluate_invariants(self.invariants, states, self.outputs, self.n_workers)
            losses = torch.concat([losses, invariant_losses], dim=1)
            print()
        self.build_normal_profile(losses)

    def build_normal_profile(self, losses):
        # best_score = np.Inf
        # best_model = None
        # for i in range(self.max_gmm_components):
        #     print("Building GMM for i =", i + 1)
        #     gmm = GaussianMixture(n_components=i + 1)
        #     gmm.fit(losses)
        #
        #     bic = gmm.bic(losses)
        #     if bic < best_score:
        #         best_score = bic
        #         best_model = gmm
        #
        # scores = best_model.score_samples(losses)
        # min_score = np.min(scores)
        # with open(self.normal_behavior_path, "w") as fd:
        #     json.dump([min_score], fd)
        # joblib.dump(best_model, self.normal_model_path)
        means = torch.mean(losses, dim=0).numpy().flatten()
        stds = torch.std(losses, dim=0).numpy().flatten()

        obj = {
            "mean": means.tolist(),
            "std": stds.tolist(),
            "loss": losses.tolist()
        }

        with open(self.normal_behavior_path, "w") as fd:
            json.dump(obj, fd)

    def compute_loss(self, unscaled_seq, scaled_seq, target):
        self.recent_outputs, self.hidden_states = self.model(unscaled_seq, scaled_seq, self.hidden_states)

        loss = []
        for loss_fn in self.loss_fns:
            losses = loss_fn(unscaled_seq, self.recent_outputs, target, self.categorical_values)
            loss.append(losses.view(1, -1))
        return torch.concat(loss, dim=1)

