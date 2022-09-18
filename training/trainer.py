import json
import joblib
import time
import os
from os import path
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, DistributedSampler
from torch import distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch import optim
from torch import multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from models import *
from utils import launch_distributed


class Trainer:
    def __init__(self, conf: dict, dataset: torch.utils.data.Dataset):
        self.conf = conf
        self.dataset = dataset
        self.optimizer = None
        self.model = None

        self.train_fraction = conf["train"]["train_fraction"]
        self.validate_fraction = conf["train"]["validate_fraction"]
        self.find_error_fraction = conf["train"]["find_error_fraction"]
        self.validation_frequency = conf["train"]["validation_frequency"]
        self.checkpoint = conf["train"]["checkpoint"]
        self.load_checkpoint = conf["train"]["load_checkpoint"]
        self.batch_size = conf["train"]["batch_size"]
        self.learning_rate = conf["train"]["lr"]
        self.decay = conf["train"]["regularization"]
        self.epochs = conf["train"]["epochs"]
        self.use_tensorboard = conf["train"]["tensorboard"]
        self.n_workers = conf["train"]['n_workers']

        self.model_path = path.join("checkpoint", self.checkpoint, "model.pt")
        self.scalar_path = path.join("checkpoint", self.checkpoint, "scaler.gz")
        self.normal_behavior_path = path.join("checkpoint", self.checkpoint, "normal_behavior.pt")
        self.results_path = path.join("results", self.checkpoint, "detection.json")

        self.loss_fn = None

    def create_prediction_model(self, train=False):
        epoch = 0
        if self.load_checkpoint or not train:
            info = torch.load(self.model_path)
            self.model = info["model"]
            self.optimizer = info["optimizer"]
            epoch = info["epoch"]
        else:
            self.model = SwatPredictionModel(self.conf)

            if self.decay > 0:
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.decay)
            else:
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        if isinstance(self.model, SwatPredictionModel):
            self.loss_fn = swat_loss
        else:
            raise RuntimeError("Unknown model type")
        return epoch

    def save_checkpoint(self, epoch):
        info = {
            "model": self.model,
            "optimizer": self.optimizer,
            "epoch": epoch
        }
        torch.save(info, self.model_path)

    def train_prediction(self):
        if self.n_workers == 1:
            self._train()
        else:
            print(f"Launching {self.n_workers} trainers")
            mp.spawn(launch_distributed, args=(self.n_workers, self._train),
                     nprocs=self.n_workers)

    def _train(self):
        train_sampler = None
        validation_sampler = None
        datalen = len(self.dataset)
        train_len = int(datalen * self.train_fraction)
        train_idx = list(range(0, train_len))
        train_data = Subset(self.dataset, train_idx)
        if not dist.is_initialized() or dist.get_rank() == 0:
            print("Train set length:", train_len)

        val_len = int(datalen * self.validate_fraction)
        val_idx = list(range(train_len, train_len + val_len))
        validation_data = Subset(self.dataset, val_idx)

        epoch = self.create_prediction_model(train=True)
        if not dist.is_initialized():
            train_data = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
            validation_data = DataLoader(validation_data)
        else:
            train_sampler = DistributedSampler(train_data, num_replicas=self.n_workers,
                                               rank=dist.get_rank(),
                                               shuffle=True)
            validation_sampler = DistributedSampler(validation_data, num_replicas=self.n_workers,
                                                    rank=dist.get_rank())
            train_data = DataLoader(train_data, batch_size=self.batch_size,
                                    shuffle=False,
                                    sampler=train_sampler)
            validation_data = DataLoader(validation_data, sampler=validation_sampler)
            self.model = DDP(self.model)

        writer = SummaryWriter("tf_board/" + self.checkpoint) if self.use_tensorboard else None
        start = time.time()
        for i in range(epoch + 1, self.epochs):
            if dist.is_initialized():
                train_sampler.set_epoch(i)
            self.model.train()
            epoch_loss = torch.tensor(0.0)
            if not dist.is_initialized() or dist.get_rank() == 0:
                loader = tqdm(train_data)
            else:
                loader = train_data
            for seq, target in loader:
                loss = self.loss_fn(self.model, seq, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            if dist.is_initialized():
                dist.reduce(epoch_loss, dst=0)

            if self.use_tensorboard:
                writer.add_scalar("Loss/train", epoch_loss.item(), i)

            if not dist.is_initialized() or dist.get_rank() == 0:
                length = time.time() - start
                print(f"Epoch {i :3d} / {self.epochs}: Loss: {epoch_loss}, {length} seconds", flush=True)
                self.save_checkpoint(i)

            if self.validation_frequency > 0 and (i + 1) % self.validation_frequency == 0:
                self.model.eval()
                val_loss = torch.tensor(0.0)
                validation_sampler.set_epoch(i)
                for seq, target in validation_data:
                    loss = self.loss_fn(self.model, seq, target)
                    val_loss += loss.detach()

                val_loss /= len(validation_data)
                if dist.is_initialized():
                    dist.reduce(val_loss, dst=0)
                    val_loss /= self.n_workers

                if self.use_tensorboard:
                    writer.add_scalar("Loss/validation", val_loss, i)

                if not dist.is_initialized() or dist.get_rank() == 0:
                    print(f"Epoch {i : 3d} Validation loss: {val_loss}", flush=True)

        if not dist.is_initialized() or dist.get_rank() == 0:
            length = time.time() - start
            print(f"Training took {length :.3f} seconds")

    def find_normal_error(self):
        start = int((self.train_fraction + self.validate_fraction) * len(self.dataset))
        size = int(self.find_error_fraction * len(self.dataset))
        idx = list(range(start, start + size))
        normal = Subset(self.dataset, idx)
        normal_data = DataLoader(normal)
        self.create_prediction_model()

        losses = []
        with torch.no_grad():
            hidden_states = [None for _ in range(len(self.conf["model"]["hidden_layers"]) - 1)]
            for features, target in tqdm(normal_data):
                loss, hidden_states = self.loss_fn(self.model, features, target, hidden_states)
                losses.append(loss)

        losses = np.array(losses)
        means = np.mean(losses)
        stds = np.std(losses)

        obj = {
            "mean": means,
            "std": stds
        }
        torch.save(obj, self.normal_behavior_path)
        print("Normal behavior saved")

    def test(self):
        dataset = DataLoader(self.dataset)
        print(f"Number of samples: {len(dataset)}")
        self.create_prediction_model()

        self.model.eval()
        obj = torch.load(self.normal_behavior_path)
        normal_means = obj["mean"]
        normal_stds = obj["std"]
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        delays = []

        attack_start = -1
        attack_detected = False

        hidden_states = [None for _ in range(len(self.conf["model"]["hidden_layers"]) - 1)]
        step = 0
        for features, target, attack in dataset:
            if attack_start > -1 and not attack:
                attack_start = -1
                if not attack_detected:
                    # Did not detect attack
                    fn += 1
                attack_detected = False

            if attack_detected:
                # Already detected attack
                continue

            features = features.unsqueeze(0)
            with torch.no_grad():
                loss, hidden_states = self.loss_fn(self.model, features, target, hidden_states)
            score = torch.abs(loss - normal_means) / normal_stds
            # print(loss, normal_means, normal_stds, attack)

            alarm = torch.any(score > 1)
            if attack_start == -1 and attack:
                attack_start = step

            if attack:
                if alarm:
                    delay = step - attack_start
                    delays.append(delay)
                    tp += 1
                    attack_detected = True
            else:
                if alarm:
                    fp += 1
                else:
                    tn += 1
            step += 1
            print(f"\r {step :5d} / {len(dataset)}: TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}", end="")

        tpr = tp / (tp + fn)
        tnr = tn / (tn + fp)
        fpr = fp / (fp + tn)
        fnr = fn / (fn + tp)

        print(f"True positive: {tpr * 100 :3.2f}")
        print(f"True negative: {tnr * 100 :3.2f}")
        print(f"False Positive: {fpr * 100 :3.2f}")
        print(f"False Negatives: {fnr * 100 :3.2f}")

        results = {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "delay": delays
        }
        with open(self.results_path, "w") as fd:
            json.dump(results, fd)
