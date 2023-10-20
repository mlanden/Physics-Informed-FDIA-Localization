import os
import sys
from os import path
from pprint import pprint
import yaml

from torch.utils.data import Subset, DataLoader
import torch.multiprocessing as mp
import numpy as np

from datasets import GridDataset
from training import PINNTrainer
from equations import build_equations


def train():
    dataset = GridDataset(conf, conf["data"]["normal"], True)
    datalen = len(dataset)
    train_len = int(datalen * train_fraction)
    train_idx = list(range(train_len))
    train_data = Subset(dataset, train_idx)
    val_len = int(datalen * validate_fraction)
    val_idx = list(range(train_len, train_len + val_len))
    validation_data = Subset(dataset, val_idx)
    
    trainer = PINNTrainer(conf, dataset)

    mp.spawn(trainer.train, args=(train_data, validation_data),
             nprocs=gpus,
             join=True)
    
def get_normal_profile():
    dataset = GridDataset(conf, conf["data"]["normal"], True)
    start = int((train_fraction + validate_fraction) * len(dataset))
    size = int(find_error_fraction * len(dataset))
    idx = list(range(start, start + size))
    normal = Subset(dataset, idx)

    trainer = PINNTrainer(conf, dataset)
    mp.spawn(trainer.create_normal_profile, args=[normal],
             nprocs=gpus,
             join=True)
    
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: main.py config")
        quit(1)

    conf_path = sys.argv[1]
    with open(conf_path, "r") as fd:
        conf = yaml.safe_load(fd)

    mp.set_start_method("spawn", force=True)
    mp.set_sharing_strategy("file_system")
    task = conf["task"]
    print("Task:", task)
    train_fraction = conf["train"]["train_fraction"]
    validate_fraction = conf["train"]["validate_fraction"]
    find_error_fraction = conf["train"]["find_error_fraction"]
    gpus = conf["train"]["gpus"]

    if task == "train":
        train()
    elif task == "error":
        get_normal_profile()
    elif task == "equ_error":
        dataset = GridDataset(conf, conf["data"]["normal"], True)
        equations = build_equations(conf, dataset.get_categorical_features(), dataset.get_continuous_features())
        features, labels = dataset.get_data()
        losses = []
        for i in range(len(features)):
            loss = equations[1].evaluate(features[i, :])
            # break
            losses.append(loss)
        print("States:", len(features))
        print("Average loss:", np.mean(losses))
        equations[1].loss_plot()