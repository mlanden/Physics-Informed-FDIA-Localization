import pickle
from functools import partial

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
from utils import save_results, make_roc_curve
eps = torch.finfo(torch.float32).eps


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
        self.profile_type = conf["model"]["profile_type"]
        self.max_gmm_components = conf["train"]["max_gmm_components"]
        self.profile_type = conf["model"]["profile_type"]
        self.load_checkpoint = conf["train"]["load_checkpoint"]

        self.checkpoint_dir = path.join(conf["train"]["checkpoint_dir"], conf["train"]["checkpoint"])
        self.results_path = path.join("results", conf["train"]["checkpoint"])
        self.normal_mean_path = path.join(self.checkpoint_dir, "normal_mean.pt")
        self.normal_gmm_path = path.join(self.checkpoint_dir, "normal_gmm.pt")

        self.normal_model_path = path.join(self.checkpoint_dir, "normal_model.gz")
        self.normal_losses_path = path.join(self.checkpoint_dir, "normal_losses.pt")
        self.invariants_path = path.join(conf["train"]["checkpoint_dir"], conf["train"]["invariants"] + "_invariants.pkl")
        self.losses_path = path.join(self.checkpoint_dir, "evaluation_losses.pt")#_invariant_std.json")
        self.eval_scores_path = path.join(self.results_path, "evaluation_losses_updated.json")
        self.anomalies = []
        self.hidden_states = None
        self.recent_outputs = None
        self.invariants = None
        self.normal_model = None
        self.normal_means = None
        self.normal_stds = None
        self.min_score = None
        self.skip_test = False

        self.loss_fns = None
        if self.loss == "invariant":
            with open(self.invariants_path, "rb") as fd:
                self.invariants = pickle.load(fd)

        self.losses = []
        self.states = []
        self.outputs = []
        self.attacks = []
        self.eval_scores = []

    def forward(self, x, hidden_states):
        return self.model(*x, hidden_states)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)  # , weight_decay=self.decay)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
        #                                                        T_max=self.max_epochs,
        #                                                        eta_min=self.learning_rate / 50)
        return [optimizer]#, [scheduler]

    def training_step(self, batch, batch_idx):
        self.hidden_states = None
        losses = self.compute_loss(*batch)
        loss = torch.mean(losses)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_validation_start(self):
        self.loss_fns = [prediction_loss]
        if self.invariants is not None:
            self.loss_fns.append(partial(invariant_loss, invariants=self.invariants))

    def validation_step(self, batch, batch_idx):
        self.hidden_states = None
        losses = self.compute_loss(*batch)
        loss = torch.mean(losses)
        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        if self.skip_test:
            return 0

        self.hidden_states = None
        return self.save_intermediates(batch)

    def on_test_start(self):
        self.loss_fns = [prediction_loss]
        if self.n_workers == 0 and self.invariants is not None:
            self.loss_fns.append(partial(invariant_loss, invariants=self.invariants))
        if path.exists(self.normal_losses_path) and self.load_checkpoint:
            self.losses = torch.load(self.normal_losses_path)
            self.skip_test = True

    def on_test_end(self):
        if not self.skip_test:
            losses = torch.concat(self.losses, dim=0)
            if self.n_workers > 0 and self.invariants is not None:
                print(f"Evaluating invariants with {self.n_workers} workers", flush=True)
                states = torch.concat(self.states, dim=0).cpu()
                invariant_losses = evaluate_invariants(self.invariants, states, self.outputs, self.n_workers)
                losses = torch.concat([losses, invariant_losses], dim=1)
                print()
            torch.save(losses, self.normal_losses_path)
        else:
            losses = torch.load(self.normal_losses_path)
        self.build_normal_profile(losses)

    def build_normal_profile(self, losses):
        print("Build normal profile", flush=True)
        losses = losses.cpu()
        # train_data = losses.numpy()[:50000, :]
        # best_score = np.Inf
        # best_model = None
        # for i in range(self.max_gmm_components):
        #     print("Building GMM for i =", i + 1)
        #     gmm = GaussianMixture(n_components=i + 1)
        #     gmm.fit(train_data)
        #
        #     bic = gmm.bic(train_data)
        #     if bic < best_score:
        #         best_score = bic
        #         best_model = gmm
        #
        # scores = best_model.score_samples(train_data)
        # min_score = np.min(scores)
        # torch.save(min_score, self.normal_gmm_path)
        # joblib.dump(best_model, self.normal_model_path)

        means = torch.mean(losses, dim=0).flatten()
        stds = torch.std(losses, dim=0).flatten()

        obj = {
            "mean": means,
            "std": stds,
            "loss": losses
        }
        torch.save(obj, self.normal_mean_path)

    def on_predict_start(self):
        self.on_test_start()
        if self.profile_type == "gmm":
            self.normal_model = joblib.load(self.normal_model_path)
            self.min_score = torch.load(self.normal_gmm_path)
        elif self.profile_type == "mean":
            obj = torch.load(self.normal_mean_path)
            self.normal_means = obj["mean"]
            self.normal_stds = obj["std"]

        if path.exists(self.losses_path) and self.load_checkpoint:
            obj = torch.load(self.losses_path)
            self.losses = obj["losses"]
            self.skip_test = True
        else:
            self.losses = []
            self.skip_test = False

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        unscaled_seq, scaled_seq, targets, attacks = batch
        self.attacks.append(attacks.cpu().detach())
        if self.skip_test:
            return 0
        self.hidden_states = None

        return self.save_intermediates((unscaled_seq, scaled_seq, targets))

    def on_predict_end(self):
        attacks = torch.concat(self.attacks, dim=0)
        if not self.skip_test:
            losses = torch.concat([loss for loss in self.losses], dim=0).cpu()
            if self.invariants is not None:
                states = torch.concat(self.states, dim=0).cpu()
                loss = evaluate_invariants(self.invariants, states, self.outputs, self.n_workers)
                losses = torch.concat([losses, loss], dim=1)
            obj = {
                "losses": losses,
                "attacks": attacks,
            }
            torch.save(obj, self.losses_path)
        else:
            losses = self.losses

        alerts = self.alert(losses, attacks)
        print("Alerts generated")
        self.detect(alerts, attacks)

        with open(self.eval_scores_path, "w") as fd:
            json.dump(self.eval_scores, fd)
        make_roc_curve(self.eval_scores_path)

    def alert(self, losses, attacks):
        if self.profile_type == "gmm":
            scores = self.normal_model.score_samples(losses.numpy())
            alarms = scores < 0.5 * self.min_score
            # if attack and not alarm:
            #     print(score / self.min_score)
            for score, attack in zip(scores, attacks):
                self.eval_scores.append((score[0] / self.min_score, attack.float().item()))
        elif self.profile_type == "mean":
            scores = torch.abs(losses - self.normal_means) / (self.normal_stds + eps)
            alarms = torch.any(scores > 74, dim=1)
            for score, alarm, attack in zip(scores, alarms, attacks):
                if alarm and not attack:
                    self.anomalies.append(torch.argmax(score).item())
                self.eval_scores.append((torch.max(score).item(), attack.float().item()))
            anomalies = set(self.anomalies)
            print(anomalies)
        else:
            raise RuntimeError("Unknown profile type")
        return alarms

    def detect(self, alerts, attacks):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        delays = []
        scores = []
        labels = []

        attack_start = -1
        attack_detected = False

        step = 0
        # self.on_evaluate_end()
        # quit()
        for alert, attack in zip(alerts, attacks):
            # if step == 5000:
            #     break
            if attack_start > -1 and not attack:
                attack_start = -1
                if not attack_detected:
                    # Did not detect attack
                    fn += 1
                attack_detected = False

            labels.append(1 if attack else 0)
            if attack_start == -1 and attack:
                attack_start = step

            if attack_detected:
                # Already detected attack
                continue
            if attack:
                if alert:
                    delay = step - attack_start
                    delays.append(delay)
                    tp += 1
                    attack_detected = True
            else:
                if alert:
                    fp += 1
                else:
                    tn += 1
            step += 1

            msg = f"{step :5d} / {len(attacks)}:"
            if tp + fn > 0:
                tpr = tp / (tp + fn)
                msg += f" TPR: {tpr * 100: 3.2f}"
            else:
                msg += " TPR: --"
            if tn + fp > 0:
                tnr = tn / (tn + fp)
                msg += f", TNR: {tnr * 100: 3.2f}"
            else:
                msg += ", TNR: --"
            if fp + tn > 0:
                fpr = fp / (fp + tn)
                msg += f", FPR: {fpr * 100: 3.2f}"
            else:
                msg += ", FPR: --"
            if fn + tp > 0:
                fnr = fn / (fn + tp)
                msg += f", FNR: {fnr * 100:3.2f}"
            else:
                msg += ", FNR: --"
            if len(delays) > 0:
                msg += f", Dwell: {np.mean(delays):.3f}"
            print(f"\r", msg, end="")
            # if step == 10000:
            #     break
        save_results(tp, tn, fp, fn, labels, self.results_path, scores, delays)

    def save_intermediates(self, batch):
        losses = self.compute_loss(*batch).detach()
        self.states.append(batch[0].cpu().detach())
        outs = []
        for out in self.recent_outputs:
            outs.append(out.cpu().detach())
        self.outputs.append(outs)
        self.losses.append(losses.cpu())
        return self.losses[-1]

    def compute_loss(self, unscaled_seq, scaled_seq, target):
        self.recent_outputs, self.hidden_states = self.model(unscaled_seq, scaled_seq, self.hidden_states)

        loss = []
        for loss_fn in self.loss_fns:
            losses = loss_fn(unscaled_seq, self.recent_outputs, target, self.categorical_values)
            loss.append(losses)

        return torch.concat(loss, dim=1)
