import os
from sched import scheduler
import sys
from os import path
from pprint import pprint
from sympy import use
import torch
import yaml

from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
import torch.multiprocessing as mp
import numpy as np
from ray import tune, air
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from datasets import GridDataset, GridGraphDataset
from training import PINNTrainer
from equations import build_equations


def train(config=None):
    if config is not None:
        conf["model"]["n_layers"] = config["n_layers"]
        conf["model"]["hidden_size"] = config["size"]
    if use_graph:
        dataset = GridGraphDataset(conf, conf["data"]["normal"], True)
    else:
        dataset = GridDataset(conf, conf["data"]["normal"], True)
    datalen = len(dataset)
    train_len = int(datalen * train_fraction)
    train_idx = list(range(train_len))
    train_data = Subset(dataset, train_idx)
    print("Training data size:", len(train_data))
    val_len = int(datalen * validate_fraction)
    val_idx = list(range(train_len, train_len + val_len))
    validation_data = Subset(dataset, val_idx)
    
    trainer = PINNTrainer(conf, dataset)

    mp.spawn(trainer.train, args=(train_data, validation_data),
             nprocs=gpus,
             join=True)
    
def get_normal_profile():
    if use_graph:
        dataset = GridGraphDataset(conf, conf["data"]["normal"], True)
    else:
        dataset = GridDataset(conf, conf["data"]["normal"], True)
    start = int((train_fraction + validate_fraction) * len(dataset))
    size = int(find_error_fraction * len(dataset))
    idx = list(range(start, start + size))
    normal = Subset(dataset, idx)
    print("Normal data size:", len(normal))

    trainer = PINNTrainer(conf, dataset)
    mp.spawn(trainer.create_normal_profile, args=[normal],
             nprocs=gpus,
             join=True)
    
def detect():
    dataset = GridDataset(conf, conf["data"]["attack"], False)
    trainer = PINNTrainer(conf, dataset)
    mp.spawn(trainer.detect, args=[dataset],
             nprocs=gpus,
             join=True)
    
def hyperparameter_optimize():
    conf["data"]["normal"] = path.abspath(conf["data"]["normal"])
    conf["data"]["ybus"] = path.abspath(conf["data"]["ybus"])
    conf["train"]["checkpoint_dir"] = path.abspath(conf["train"]["checkpoint_dir"])

    config = {
        "n_layers": tune.choice([i for i in range(2, 9)]),
        "size": tune.choice([2 ** i for i in range(9)])
    }

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=conf["train"]["epochs"],
        grace_period=1,
        reduction_factor=2
    )

    reporter = CLIReporter(parameter_columns=list(config.keys()),
                           metric_columns=["loss", "training_iterations"])
    
    results = tune.run(
        train,
        resources_per_trial={"gpu": 1},
        config=config,
        num_samples=conf["train"]["num_samples"],
        progress_reporter=reporter,
        run_config=air.RunConfig(name="tune_ics",
                                 progress_reporter=reporter),
        scheduler=scheduler
    )
    best_trial = results.get_best_trial("loss", "min")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial validation loss: {best_trial.last_result['loss']}")

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
    use_graph = conf["model"]["graph"]

    if task == "train":
        train()
    elif task == "error":
        get_normal_profile()
    elif task == "test":
        detect()
    elif task == "hyperparameter_optimize":
        hyperparameter_optimize()
    elif task == "equ_error":
        losses = []
        if use_graph:
            dataset = GridGraphDataset(conf, conf["data"]["normal"], True)
            equations = build_equations(conf, dataset.get_categorical_features(), dataset.get_continuous_features())
            loader = DataLoader(dataset, batch_size=conf["train"]["batch_size"])
            for batch in loader:
                for equ in equations:
                    loss = equ.evaluate(batch)
                    losses.append(loss)
            loss = torch.cat(losses)
            print("average loss:", torch.mean(loss))
            print("Standard dev:", torch.std(loss))
        else:
            dataset = GridDataset(conf, conf["data"]["normal"], True)
            equations = build_equations(conf, dataset.get_categorical_features(), dataset.get_continuous_features())
            features, labels = dataset.get_data()
            start = 0
            end = conf["train"]["batch_size"]
            while start < len(features):
                batch = features[start: end, :]
                for equ in equations:
                    loss = equ.evaluate(batch)
                    losses.append(loss)
                
                start = end
                end += conf["train"]["batch_size"]
                if end > len(features):
                    end = len(features)
            losses = np.concatenate(losses)
            print("States:", len(features))
            print("Average loss:", np.mean(losses))
            for i in range(len(equations)):
                equations[i].loss_plot()
    else:
        raise RuntimeError("Unknown task")