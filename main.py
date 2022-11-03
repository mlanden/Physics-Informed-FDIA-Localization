import os
from os import path
import yaml
import sys
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from torch.utils.data import Subset, DataLoader

from datasets import SWATDataset
from training import hyperparameter_optimize, ICSTrainer
from evaluation import NNEvaluator
from invariants import generate_predicates, InvariantMiner


def train():
    
    trainer = Trainer(default_root_dir=checkpoint_dir,
                      log_every_n_steps=9,
                      max_epochs=conf["train"]["epochs"],
                      devices=conf["train"]["n_workers"],
                      accelerator="gpu" if torch.cuda.is_available() else "cpu",
                      callbacks=[ModelCheckpoint(dirpath=checkpoint_dir,
                                                 filename=checkpoint,
                                                 save_last=True,
                                                 every_n_train_steps=0,
                                                 every_n_epochs=1,
                                                 save_on_train_epoch_end=True),
                                 RichProgressBar(leave=True)])
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
    if load_checkpoint:
        trainer.fit(model, train_loader, val_loader,
                    ckpt_path=checkpoint_to_load)
    else:
        trainer.fit(model, train_loader, val_loader)


def find_normal_error():
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    trainer = Trainer(default_root_dir=checkpoint_dir)

    dataset = SWATDataset(conf, conf["data"]["normal"],
                          sequence_len=1,
                          train=True,
                          load_scaler=True)
    model = ICSTrainer.load_from_checkpoint(checkpoint_to_load, conf=conf, map_location=device)
    start = int((train_fraction + validate_fraction) * len(dataset))
    size = int(find_error_fraction * len(dataset))
    idx = list(range(start, start + size))
    normal = Subset(dataset, idx)
    loader = DataLoader(normal)

    trainer.test(model, loader)


def test():
    dataset = SWATDataset(conf, conf["data"]["attack"],
                          sequence_len=1,
                          train=False,
                          load_scaler=True)
    type_ = conf["train"]["type"]
    if type_ == "prediction":
        model = ICSTrainer.load_from_checkpoint(checkpoint_to_load, conf=conf)
        evaluator = NNEvaluator(conf, model, dataset)
        evaluator.evaluate()
    elif type_ == "invariants":
        miner = InvariantMiner(conf, dataset)
        miner.evaluate()
    else:
        raise RuntimeError("Unknown evaluation type")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: main.py config")
        quit(1)

    conf_path = sys.argv[1]
    with open(conf_path, "r") as fd:
        conf = yaml.safe_load(fd)
    checkpoint = conf["train"]["checkpoint"]
    # checkpoint_to_load = "/home/mlanden/ICS-Attack-Detection/checkpoint/swat_2015_full/swat_2015_full-v1.ckpt"

    checkpoint_dir = path.join("checkpoint", checkpoint)
    checkpoint_to_load = path.join(checkpoint_dir, "last-v4.ckpt")
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
    load_checkpoint = conf["train"]["load_checkpoint"]
    checkpoint_path = checkpoint_dir + ".ckpt"
    find_error_fraction = conf["train"]["find_error_fraction"]

    if task == "train":
        train()
    elif task == "hyperparameter_optimize":
        dataset = SWATDataset(conf, conf["data"]["normal"],
                              sequence_len=conf["model"]["sequence_length"],
                              train=True,
                              load_scaler=False)
        hyperparameter_optimize(conf, dataset)
    elif task == "error":
        find_normal_error()
    elif task == "test":
        test()
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
