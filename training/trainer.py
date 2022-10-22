import json
import pickle
import signal

import joblib
import time
import os
from os import path
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, DistributedSampler
from torch import distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch import optim
from torch import multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from models import *
from datasets import *
from utils import launch_distributed


class Trainer:
    def __init__(self, conf: dict, dataset: ICSDataset):
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
        self.invariants_path = path.join("checkpoint", self.checkpoint, "invariants.pkl")
        self.results_path = path.join("results", self.checkpoint)
        self.loss = conf["train"]["loss"]
        self.invariants = None
        if self.loss == "invariant":
            with open(self.invariants_path, "rb") as fd:
                self.invariants = pickle.load(fd)
        self.loss_fns = get_losses(self.dataset, self.invariants)

    def create_prediction_model(self, train=False):
        epoch = 0
        if self.load_checkpoint or not train:
            info = torch.load(self.model_path)
            self.model = info["model"]
            self.optimizer = info["optimizer"]
            epoch = info["epoch"]
        else:
            self.model = PredictionModel(self.conf, self.dataset)

            if self.decay > 0:
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.decay)
            else:
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        return epoch

    def save_checkpoint(self, epoch):
        if isinstance(self.model, DDP):
            model = self.model.module
        else:
            model = self.model
        info = {
            "model": model,
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
        signal.signal(signal.SIGINT, quit)
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
        for i in range(epoch, self.epochs):
            if dist.is_initialized():
                train_sampler.set_epoch(i)
            self.model.train()
            epoch_loss = torch.tensor(0.0)

            if not dist.is_initialized() or dist.get_rank() == 0:
                loader = tqdm(train_data)
                print(f"Epoch {i :3d} / {self.epochs}", flush=True)
            else:
                loader = train_data
            for seq, target in loader:
                losses = self.compute_loss(seq, target)
                loss = torch.sum(losses, dim=1)

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
                print(f"Loss: {epoch_loss / len(loader)}, {length} seconds", flush=True)
                self.save_checkpoint(i + 1)

            if self.validation_frequency > 0 and i % self.validation_frequency == 0:
                self.model.eval()
                val_loss = torch.tensor(0.0)
                if validation_sampler is not None:
                    validation_sampler.set_epoch(i)
                for seq, target in validation_data:
                    losses = self.compute_loss(seq, target)
                    loss = torch.sum(losses)
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

    def compute_loss(self, seq, target, hidden_states=None):
        loss = []
        for loss_fn in self.loss_fns:
            losses = loss_fn(self.model, seq, target, self.dataset.get_categorical_features(), hidden_states)
            loss.append(losses.view(1, -1))
        return torch.concat(loss, dim=1)

    def find_normal_error(self):
        start = int((self.train_fraction + self.validate_fraction) * len(self.dataset))
        size = int(self.find_error_fraction * len(self.dataset))
        idx = list(range(start, start + size))
        normal = Subset(self.dataset, idx)
        normal_data = DataLoader(normal)
        self.create_prediction_model()

        loss_vectors = []
        with torch.no_grad():
            hidden_states = [None for _ in range(len(self.conf["model"]["hidden_layers"]) - 1)]
            for features, target in tqdm(normal_data):
                losses = self.compute_loss(features, target, hidden_states)

                loss_vectors.append(losses)

        loss_vectors = torch.concat(loss_vectors, 0)
        means = torch.mean(loss_vectors, 0).numpy().flatten()
        stds = torch.std(loss_vectors, 0).numpy().flatten()
        print(means)
        print(stds)

        self.plot_loss_dist(means, stds)
        obj = {
            "mean": means,
            "std": stds
        }
        torch.save(obj, self.normal_behavior_path)
        print("Normal behavior saved")

        # fpr, tpr, thresholds = roc_curve(labels, scores)
        #
        # fig, ax = plt.subplots()
        #
        # ax.plot(fpr, tpr)
        # ax.set(xlabel="False Positive Rate",
        #        ylabel="True Positive Rate",
        #        title="ROC Curve")
        # plt.savefig(path.join(self.results_path, "roc.png"))

    def plot_loss_dist(self, means, stds):
        fig, (mean_ax, std_ax) = plt.subplots(1, 2)
        mean_ax.hist(means)
        std_ax.hist(stds)
        mean_ax.set(title="Mean Losses",
                    ylabel="Mean Loss")
        std_ax.set(title="Standard Deviation Losses",
                   ylabel="Standard Deviation Loss")
        plt.tight_layout()
        plt.savefig(path.join(self.results_path, "hists.png"))
