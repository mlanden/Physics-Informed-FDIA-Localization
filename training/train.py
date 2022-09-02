from os import path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from models import PredictionModel


def rnn_train(conf, dataset):
    checkpoint = conf["train"]["checkpoint"]
    normal_data = DataLoader(dataset, batch_size = conf["train"]["batch_size"], shuffle = True)

    model = PredictionModel(conf)
    print(model)
    decay = conf["train"]["regularization"]
    if decay > 0:
        optimizer = optim.Adam(model.parameters(), lr = conf["train"]["lr"], weight_decay = decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr = conf["train"]["lr"])

    checkpoint = path.join("checkpoint", checkpoint, "model.pt")
    loss_fn = torch.nn.MSELoss()
    epochs = conf["train"]["epochs"]
    writer = SummaryWriter("tf_board/Swat_prediction")
    for i in range(epochs):
        epoch_loss = 0
        for seq, target in normal_data:
            predicted = model(seq)
            predicted = predicted[:, -1, :]

            loss = loss_fn(predicted, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        writer.add_scalar("Loss/train", epoch_loss, i)

        if i % 10 == 0:
            print(f"Epoch {i :3d} / {epochs}: Loss: {epoch_loss}")
            torch.save(model, checkpoint)


def find_normal_error(conf, dataset):
    checkpoint = conf["train"]["checkpoint"]
    checkpoint = path.join("checkpoint", checkpoint, "model.pt")
    normal_data = DataLoader(dataset)

    model = torch.load(checkpoint)
    model.eval()
    errors = []

    hidden_state = [None for _ in range(len(conf["model"]["hidden_layers"]) - 1)]
    for i, (features, target) in enumerate(normal_data):
        predicted, hidden_state = model(features, hidden_state)
        predicted = predicted.view(1, -1)
        target = target.view(1, -1)

        error = torch.abs(predicted - target)
        errors.append(error)

    errors = torch.concat(errors, dim = 0)
    means = torch.mean(errors, dim = 0)
    stds = torch.std(errors, dim = 0)
    print(means, stds)
    # print(f"Mean: {mean}, Standard deviation: {std}.")
