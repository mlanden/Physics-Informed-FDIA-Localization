import os
from sched import scheduler
import sys
from os import path
from pprint import pprint
from ray import is_initialized
from sympy import use
import torch
import yaml

from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
import torch.multiprocessing as mp
import numpy as np

import ray
from ray import tune
from ray.train import ScalingConfig, RunConfig, CheckpointConfig
from ray.train.torch import TorchTrainer
from ray.tune import CLIReporter
from ray.tune.tune_config import TuneConfig
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining

from datasets import GridDataset, GridGraphDataset
from training import PINNTrainer
from equations import build_equations
from utils import generate_fdia

def train(config=None):
    if config is not None:
        conf["model"]["n_heads"] = config.get("n_heads", conf["model"]["n_heads"])
        conf["model"]["hidden_size"] = config.get("size", conf["model"]["hidden_size"])
        conf["train"]["lr"] = config["lr"]
        conf["train"]["regularization"] = config["regularization"]

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
    if ray.is_initialized():
        trainer.train(0, train_data, validation_data)
    else:
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
    if use_graph:
        dataset = GridGraphDataset(conf, conf["data"]["attack"], False)
    else:
        dataset = GridDataset(conf, conf["data"]["attack"], False)
    trainer = PINNTrainer(conf, dataset)
    mp.spawn(trainer.detect, args=[dataset],
             nprocs=gpus,
             join=True)
    
def hyperparameter_optimize():
    conf["data"]["normal"] = path.abspath(conf["data"]["normal"])
    conf["data"]["ybus"] = path.abspath(conf["data"]["ybus"])
    conf["data"]["types"] = path.abspath(conf["data"]["types"])
    conf["train"]["checkpoint_dir"] = path.abspath(conf["train"]["checkpoint_dir"])
    population_training = conf["train"].get("population_train", 1)
    config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "dropout": tune.uniform(0.1, 0.8),
        "regularization": tune.loguniform(1e-5, 1e-1)
    }

    if not population_training:
        config.update({
            "n_heads": tune.choice([i for i in range(2, 9)]),
            "size": tune.choice([2 ** i for i in range(9)]),
        })
    if not population_training:
        scheduler = ASHAScheduler(
            max_t=conf["train"]["epochs"],
            grace_period=1,
            metric="loss",
            mode="min",
            reduction_factor=2
        )
    else:
        scheduler = PopulationBasedTraining(
            time_attr="training_iteration",
            perturbation_interval=5,
            metric="loss",
            mode="min",
            hyperparam_mutations={
                "train_loop_config": config
            }
        )
    
    trainer = TorchTrainer(
        train,
        scaling_config=ScalingConfig(num_workers=1, use_gpu=conf["train"]["cuda"]))
    
    if conf["train"]["load_checkpoint"]:
        tuner = tune.Tuner.restore(
            conf["train"]["checkpoint_dir"],
            trainer
        )
    else:
        tuner = tune.Tuner(
        trainer,
        param_space={
            "train_loop_config": config
        },
        tune_config=TuneConfig(
            num_samples=conf["train"]["num_samples"],
            scheduler=scheduler
        ),
        run_config=RunConfig(storage_path=path.abspath("./checkpoint"),
                             checkpoint_config=CheckpointConfig(num_to_keep=4, 
                                                                checkpoint_score_attribute="loss",
                                                                checkpoint_score_order="min")),

    )
    
    results = tuner.fit()
    best_trial = results.get_best_result("loss", "min")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial checkpoint: {best_trial.path}")

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
    elif task == "attack_generation":
        generate_fdia(conf)
    else:
        raise RuntimeError("Unknown task")