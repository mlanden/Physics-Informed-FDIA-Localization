import os
from os import path
import yaml
import sys
import torch
from pytorch_lightning import Trainer
from torch.utils.data import Subset, DataLoader

from datasets import SWATDataset
from training import hyperparameter_optimize, ICSTrainer
from evaluation import NNEvaluator
from invariants import generate_predicates, InvariantMiner

def train():
    trainer = Trainer()

    dataset = SWATDataset(conf, conf["data"]["normal"],
                          sequence_len=conf["model"]["sequence_length"],
                          train=True,
                          load_scaler=False)
    datalen = len(dataset)
    train_len = int(datalen * train_fraction)
    train_idx = list(range(0, train_len))
    train_data = Subset(dataset, train_idx)
    val_len = int(datalen * validate_fraction)
    val_idx = list(range(train_len, train_len + val_len))
    validation_data = Subset(dataset, val_idx)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False, drop_last=False)
    model = ICSTrainer(conf, dataset.get_categorical_features())
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: main.py config")
        quit(1)

    conf_path = sys.argv[1]
    with open(conf_path, "r") as fd:
        conf = yaml.safe_load(fd)
    checkpoint_dir = path.join("checkpoint", conf["train"]["checkpoint"])
    results_dir = path.join("results", conf["train"]["checkpoint"])
    if not path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    if not path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    task = conf["task"]
    print("Task:", task)
    train_fraction = conf["train"]["train_fraction"]
    validate_fraction = conf["train"]["validate_fraction"]
    batch_size = conf["train"]["batch_size"]
    checkpoint = conf["train"]["checkpoint"]
    load_checkpoint = conf["train"]["load_checkpoint"]

    if task == "train":
        train()
    elif task == "hyperparameter_optimize":
        dataset = SWATDataset(conf, conf["data"]["normal"],
                              sequence_len=conf["model"]["sequence_length"],
                              train=True,
                              load_scaler=False)
        hyperparameter_optimize(conf, dataset)
    elif task == "error":
        dataset = SWATDataset(conf, conf["data"]["normal"],
                              sequence_len=1,
                              train=True,
                              load_scaler=True)
        trainer = Trainer(conf, dataset)
        trainer.find_normal_error()
    elif task == "test":
        dataset = SWATDataset(conf, conf["data"]["attack"],
                              sequence_len=1,
                              train=False,
                              load_scaler=True)
        type_ = conf["train"]["type"]
        if type_ == "prediction":
            evaluator = NNEvaluator(conf, dataset)
            evaluator.evaluate()
        elif type_ == "invariants":
            miner = InvariantMiner(conf, dataset)
            miner.evaluate()
        else:
            raise RuntimeError("Unknown evaluation type")

    elif task == "predicates":
        dataset = SWATDataset(conf, conf["data"]["normal"],
                              sequence_len=conf["model"]["sequence_length"],
                              train=True,
                              load_scaler=False)
        predicates = generate_predicates(dataset, conf)
    elif "invariants":
        dataset = SWATDataset(conf, conf["data"]["normal"],
                              sequence_len=conf["model"]["sequence_length"],
                              train=True,
                              load_scaler=False)
        miner = InvariantMiner(conf, dataset)
        miner.mine_invariants()
    else:
        raise RuntimeError(f"Unknown task: {task}")
