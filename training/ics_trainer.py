import pickle
from functools import partial

import joblib
import json
from os import path
from sklearn.metrics import confusion_matrix
from collections import defaultdict
import numpy as np
from sklearn.mixture import GaussianMixture

import torch
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from torch import optim

from models import PredictionModel, AutoencoderModel, invariant_loss, prediction_loss, equation_loss
from utils import evaluate_loss
from equations import build_equations
eps = torch.finfo(torch.float32).eps


class ICSTrainer(LightningModule):

    def __init__(self, conf, categorical_values, continuous_values):
        super(ICSTrainer, self).__init__()
        self.save_hyperparameters()
        self.conf = conf
        self.categorical_values = categorical_values
        self.continuous_values = continuous_values
        self.learning_rate = conf["train"]["lr"]
        self.decay = conf["train"]["regularization"]
        self.loss = conf["train"]["loss"]
        self.n_workers = conf["train"]['n_workers']
        self.max_epochs = conf["train"]["epochs"]
        self.profile_type = conf["model"]["profile_type"]
        self.max_gmm_components = conf["train"]["max_gmm_components"]
        self.profile_type = conf["model"]["profile_type"]
        self.load_checkpoint = conf["train"]["load_checkpoint"]
        self.scale = conf["train"]["scale"]
        self.model_type = conf["model"]["type"]

        self.checkpoint_dir = path.join(conf["train"]["checkpoint_dir"], conf["train"]["checkpoint"])
        self.results_path = path.join("results", conf["train"]["checkpoint"])
        self.normal_mean_path = path.join(self.checkpoint_dir, "normal_mean.pt")
        self.normal_gmm_path = path.join(self.checkpoint_dir, "normal_gmm.pt")

        self.normal_model_path = path.join(self.checkpoint_dir, "normal_model.gz")
        self.normal_losses_path = path.join(self.checkpoint_dir, "normal_losses.pt")
        self.invariants_path = path.join(conf["train"]["checkpoint_dir"], conf["train"]["invariants"] + "_invariants.pkl")
        self.losses_path = path.join(self.checkpoint_dir, "evaluation_losses.pt")#_invariant_std.json")
        self.eval_scores_path = path.join(self.results_path, "evaluation_losses.json")
        self.recent_outputs = None
        self.invariants = None
        self.equations = None
        self.normal_model = None
        self.normal_means = None
        self.normal_stds = None
        self.min_score = None
        self.skip_test = False
        self.loss_fns = None
        self.loss_names = None

        if self.model_type == "prediction":
            self.model = PredictionModel(conf, categorical_values)
        elif self.model_type == "autoencoder":
            self.model = AutoencoderModel(conf, categorical_values)
        else:
            raise RuntimeError("Unknown model type")
        
        if self.loss == "invariant":
            with open(self.invariants_path, "rb") as fd:
                self.invariants = pickle.load(fd)
            anteceds = []
            consequents = []
            for i in self.invariants:
                anteceds.append(len(i.antecedent))
                consequents.append(len(i.consequent))
            print("Mean antecedent:", np.mean(anteceds))
            print("Mean consequent:", np.mean(consequents))
        elif self.loss == "equation":
            self.equations = build_equations(conf, categorical_values, continuous_values)

        self.losses = []
        self.states = []
        self.batch_ids = []
        self.outputs = []
        self.attacks = []
        self.eval_scores = []

    def forward(self, *x):
        # print(x)
        return self.model(*x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.decay)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
        #                                                        T_max=self.max_epochs,
        #                                                        eta_min=self.learning_rate / 50)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=15)
        # return [optimizer], [scheduler]
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

    def training_step(self, batch, batch_idx):
        losses = self._compute_combine_losses(batch)

        for i in range(len(self.loss_names)):
            self.log(self.loss_names[i] + "_Train_loss", losses[i], prog_bar=True, on_step=False,
                     on_epoch=True, sync_dist=False)

        loss = torch.sum(losses)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=False)
        return loss

    def _compute_combine_losses(self, batch):
        losses = self.compute_loss(*batch)
        for i in range(len(losses)):
            losses[i] = torch.mean(losses[i], dim=1).view(-1, 1)
        losses = torch.cat(losses, dim=1)
        losses = torch.mean(losses, dim=0)
        return losses

    def on_validation_start(self):
        self.loss_fns = [prediction_loss]
        self.loss_names = ["Prediction"]
        if self.invariants is not None:
            self.loss_fns.append(partial(invariant_loss, invariants=self.invariants))
            self.loss_names.append("Invariant")
        if self.equations is not None:
            self.loss_fns.append(partial(equation_loss, equations=self.equations))
            self.loss_names.append("Equation")

    def validation_step(self, batch, batch_idx):
        losses = self._compute_combine_losses(batch)
        loss = torch.sum(losses)
        self.log("val_loss", loss, prog_bar=True, sync_dist=False, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        if self.skip_test:
            return 0

        return self.save_intermediates(batch)

    def on_test_start(self):
        self.loss_fns = [prediction_loss]
        if self.n_workers == 0 and self.invariants is not None:
            self.loss_fns.append(partial(invariant_loss, invariants=self.invariants))
        if path.exists(self.normal_losses_path) and self.load_checkpoint:
            self.losses = torch.load(self.normal_losses_path)
            self.skip_test = True

    def save_intermediates(self, batch):
        losses = self.compute_loss(*batch)
        for i in range(len(losses)):
            losses[i] = torch.mean(losses[i], dim=1).view(-1, 1)
        losses = torch.cat(losses, dim=1).detach()
        self.states.append(batch[0].cpu().detach())
        outs = []
        for out in self.recent_outputs:
            outs.append(out.cpu().detach())
        self.outputs.append(outs)
        self.losses.append(losses.cpu())
        return self.losses[-1]

    def compute_loss(self, unscaled_seq, scaled_seq, target):
        self.recent_outputs = self.model(unscaled_seq, scaled_seq)

        loss = []
        for i, loss_fn in enumerate(self.loss_fns):
            losses = self.scale[i] * loss_fn(unscaled_seq, self.recent_outputs, target, self.categorical_values)
            loss.append(losses)
            # print(losses)
        # return torch.concat(loss, dim=1)
        return loss

    def on_test_end(self):
        if not self.skip_test:
            losses = torch.concat(self.losses, dim=0)
            losses = self._complete_losses(losses)
        else:
            losses = torch.load(self.normal_losses_path)
        if self.global_rank == 0:
            self.build_normal_profile(losses)

    def _complete_losses(self, losses):
        if self.trainer.gpus != 1:
            losses = self.all_gather(losses)
            losses = losses.view(losses.shape[0] * losses.shape[1], -1)
        if self.n_workers > 0 and (self.invariants is not None or self.equations is not None):
            combined_outputs, states = self._prep_evaluate_invariants()
            if self.global_rank == 0:
                print(f"Evaluating losses with {self.n_workers} workers", flush=True)
                losses = losses.cpu()
                if self.invariants is not None:
                    object_loss = evaluate_loss(self.invariants, states, combined_outputs, self.n_workers)
                elif self.equations is not None:
                    object_loss = evaluate_loss(self.equations, states, combined_outputs, self.n_workers)
                # object_loss = torch.mean(object_loss, dim=1).view(-1, 1)
                losses = torch.concat([losses, object_loss], dim=1)
        torch.save(losses, self.normal_losses_path)
        return losses

    def _prep_evaluate_invariants(self):
        states = torch.concat(self.states, dim=0)
        combined_outputs = []
        for i in range(len(self.outputs[0])):
            outs = [self.outputs[place][i].cpu().detach() for place in range(len(self.outputs))]
            torch_output = torch.cat(outs, dim=0)
            combined_outputs.append(torch_output)
        if self.trainer.gpus != 1:
            states = self.all_gather(states)
            states = states.view(states.shape[0] * states.shape[1], states.shape[2], -1)
            for i in range(len(combined_outputs)):
                combined_outputs[i] = self.all_gather(combined_outputs[i])
                combined_outputs[i] = combined_outputs[i].view(combined_outputs[i].shape[0] *
                                                               combined_outputs[i].shape[1], -1)
                # print(i, combined_outputs[i].shape, flush=True)
        return combined_outputs, states

    def build_normal_profile(self, losses):
        print("Build normal profile", flush=True)
        # train_data = losses.numpy()
        # np.random.shuffle(train_data)
        # train_data = train_data[:50000]
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
            self.normal_means = obj["mean"].cpu()
            self.normal_stds = obj["std"].cpu()
            if self.global_rank == 0:
                print("Mean normal profile:", self.normal_means)
                print("Standard deviation of normal profile:", self.normal_stds)

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

        return self.save_intermediates((unscaled_seq, scaled_seq, targets))

    def on_predict_end(self):
        attacks = torch.concat(self.attacks, dim=0)
        if self.trainer.gpus != 1:
            attacks = self.all_gather(attacks)
            attacks = attacks.view(attacks.shape[0] * attacks.shape[1], -1).cpu()
        if not self.skip_test:
            losses = torch.concat([loss for loss in self.losses], dim=0)
            losses = self._complete_losses(losses)
            if self.global_rank == 0:
                obj = {
                    "losses": losses,
                    "attacks": attacks,
                }
                torch.save(obj, self.losses_path)
        else:
            losses = self.losses

        if self.global_rank == 0:
            alerts = self.alert(losses, attacks)
            print("Alerts generated")
            self.detect(alerts, attacks)

            with open(self.eval_scores_path, "w") as fd:
                json.dump(self.eval_scores, fd)

    def alert(self, losses, attacks):
        losses = losses.cpu()
        if self.profile_type == "gmm":
            print("Min", self.min_score)
            scores = self.normal_model.score_samples(losses.numpy())
            alarms = scores < 1.14 * self.min_score
            # if attack and not alarm:
            #     print(score / self.min_score)
            for score, attack in zip(scores, attacks):
                self.eval_scores.append((-score / self.min_score, attack.float().item()))
        elif self.profile_type == "mean":
            scores = torch.abs(losses - self.normal_means) / (self.normal_stds + eps)
            debug = []
            # alarms = torch.any(scores > 7, dim=1)
            alarms = torch.max(scores, dim=1).values > 1.5
            for score, alarm, attack in zip(scores, alarms, attacks):
                # if not alarm and attack:
                #     print(torch.max(score))
                #     self.anomalies[torch.argmax(score).item()] += 1
                if alarm:
                    debug.append((torch.max(score).item(), torch.argmax(score).item(), attack.item()))
                self.eval_scores.append((torch.max(score).item(), attack.float().item()))

            with open("debug.json", "w") as fd:
                json.dump(debug, fd)
            # anomalies = dict(sorted(self.anomalies.items(), key=lambda x: x[1], reverse=True))
            # for i in anomalies:
            #     print(i, anomalies[i])
            #     if i >= 51:
            #         inv = self.invariants[i - 51]
            #         print(inv)
            #         print()
        else:
            raise RuntimeError("Unknown profile type")
        return alarms

    def detect(self, predicted, actual):
        cm = confusion_matrix(actual, predicted)
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
        accuracy = (tp + tn) / len(actual)
        print(f"True Positive: {tpr * 100 :3.2f}")
        print(f"True Negative: {tnr * 100 :3.2f}")
        print(f"False Positive: {fpr * 100 :3.2f}")
        print(f"False Negatives: {fnr * 100 :3.2f}")
        print(f"F1 Score: {f1 * 100 :3.2f}")
        print(f"Precision: {precision * 100 :3.2f}")
        print(f"Accuracy: {accuracy * 100 :3.2f}")

        dwells = []
        fns = []
        attack = False
        detect = False
        start = 0
        for i in range(len(actual)):
            if actual[i] and not predicted[i]:
                fns.append(i)
            if actual[i] and not attack:
                attack = True
                start = i

            if attack and not detect and predicted[i]:
                dwells.append(i - start)
                attack = False
                detect = True
        dwell_time = np.mean(dwells)
        print(f"Dwell time: {dwell_time :.2f}")

        with open("fn_ids_physics_informed.json", "w") as fd:
            json.dump(fns, fd)
