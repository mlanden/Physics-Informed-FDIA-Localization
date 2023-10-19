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


def train():
    dataset = GridDataset(conf, conf["data"]["normal"], conf["model"]["window_size"], True)
    datalen = len(dataset)
    train_len = int(datalen * train_fraction)
    train_idx = list(range(train_len))
    train_data = Subset(dataset, train_idx)
    val_len = int(datalen * validate_fraction)
    val_idx = list(range(train_len, train_len + val_len))
    validation_data = Subset(dataset, val_idx)
    
    trainer = PINNTrainer(conf, dataset)

    gpus = conf["train"]["gpus"]
    mp.spawn(trainer.train, args=(train_data, validation_data),
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

    if task == "train":
        train()