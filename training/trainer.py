import time
from os import path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from models import PredictionModel


class Trainer:
    def __init__(self, conf, dataset):
        self.conf = conf
        self.dataset = dataset

        self.train_fraction = conf["train"]["train_fraction"]
        self.validate_fraction = conf["train"]["validate_fraction"]
        self.find_error_fraction = conf["train"]["find_error_fraction"]
        self.validation_frequency = conf["train"]["validation_frequency"]
        self.checkpoint = conf["train"]["checkpoint"]
        self.batch_size = conf["train"]["batch_size"]
        self.learning_rate = conf["train"]["lr"]
        self.decay = conf["train"]["regularization"]
        self.epochs = conf["train"]["epochs"]
        self.use_tensorboard = conf["train"]["tensorboard"]
        self.model_path = path.join("checkpoint", self.checkpoint, "model.pt")
        self.normal_behavior_path = path.join("checkpoint", self.checkpoint, "normal_behavior.pt")

    def train_prediction(self):
        datalen = len(self.dataset)
        train_len = int(datalen * self.train_fraction)
        train_idx = list(range(0, train_len))
        train_data = Subset(self.dataset, train_idx)

        val_len = int(datalen * self.validate_fraction)
        val_idx = list(range(train_len, train_len + val_len))
        validation_data = Subset(self.dataset, val_idx)

        train_data = DataLoader(train_data, batch_size = self.batch_size, shuffle = True)
        validation_data = DataLoader(validation_data)
        print(f"Training sequences: {train_len}, Validation sequences: {val_len}")

        model = PredictionModel(self.conf)

        if self.decay > 0:
            optimizer = optim.Adam(model.parameters(), lr = self.learning_rate, weight_decay = self.decay)
        else:
            optimizer = optim.Adam(model.parameters(), lr = self.learning_rate)

        loss_fn = torch.nn.MSELoss()

        writer = SummaryWriter("tf_board/Swat_prediction") if self.use_tensorboard else None
        start = time.time()
        for i in range(self.epochs):
            model.train()
            epoch_loss = 0
            for seq, target in train_data:
                predicted = model(seq)
                predicted = predicted[:, -1, :]

                loss = loss_fn(predicted, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            if self.use_tensorboard:
                writer.add_scalar("Loss/train", epoch_loss, i)

            if (i + 1) % self.validation_frequency == 0:
                model.eval()
                val_loss = 0
                for seq, target in validation_data:
                    predicted = model(seq)
                    predicted = predicted[:, -1, :]
                    val_loss += loss_fn(predicted, target)

                val_loss /= val_len
                if self.use_tensorboard:
                    writer.add_scalar("Loss/validation", val_loss, i)

                print(f"Epoch {i : 3d} Validation loss: {val_loss}")

            if i % 10 == 0:
                print(f"Epoch {i :3d} / {self.epochs}: Loss: {epoch_loss}")
                torch.save(model, self.model_path)

        length = time.time() - start
        print(f"Training took {length / 1000 :.3f} seconds")

    def find_normal_error(self):
        start = int((self.train_fraction + self.validate_fraction) * len(self.dataset))
        size = int(self.find_error_fraction * len(self.dataset))
        idx = list(range(start, start + size))
        normal = Subset(self.dataset, idx)
        normal_data = DataLoader(normal)
        print(f"Normal sequences: {len(normal_data)}")

        model = torch.load(self.model_path)
        model.eval()
        errors = []

        hidden_state = [None for _ in range(len(self.conf["model"]["hidden_layers"]) - 1)]
        for features, target in normal_data:
            predicted, hidden_state = model(features, hidden_state)
            predicted = predicted.view(1, -1)
            target = target.view(1, -1)

            error = torch.abs(predicted - target)
            errors.append(error)

        errors = torch.concat(errors, dim = 0)
        means = torch.mean(errors, dim = 0)
        stds = torch.std(errors, dim = 0)

        obj = {
            "mean": means,
            "std": stds
        }
        torch.save(obj, self.normal_behavior_path)
        print("Normal behavior saved")

    # def test(self):
