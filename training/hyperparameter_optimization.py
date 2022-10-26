import sys
import logging
import os.path
from functools import partial
import os
import optuna
from torch import distributed as dist
from .trainer import Trainer
import torch.multiprocessing as mp


def train(trial: optuna.Trial, conf, dataset):
    if conf["train"]["n_workers"] > 1:
        trial = optuna.integration.TorchDistributedTrial(trial)
        conf["train"]["n_workers"] = 1

    print("Train", dist.get_rank(), flush=True)
    learning_rate = trial.suggest_float("Learning rate", 1e-5, 1e-2, log=True)
    batch_size = 2 ** trial.suggest_int("batch_size", 5, 8)
    conf["train"]["lr"] = learning_rate
    conf["train"]["batch_size"] = batch_size

    trainer = Trainer(conf, dataset)
    return trainer.train()


def get_study():
    study = optuna.create_study(pruner=optuna.pruners.MedianPruner())
    return study


def hyperparameter_optimize(conf: dict, dataset):
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    n_workers = conf["train"]["n_workers"]
    n_trials = 5

    if n_workers > 1:
        mp.spawn(launch_study, args=(conf, dataset, n_trials, n_workers), nprocs=n_workers)
    else:
        study = get_study()
        study.optimize(partial(train, conf=conf, dataset=dataset), n_trials=2)
        best_parameters = study.best_params
        print(best_parameters)


def launch_study(rank, conf, dataset, n_trials, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(dist.Backend.GLOO, rank=rank, world_size=world_size)
    study = None
    if rank == 0:
        print("Rank 0")
        study = get_study()
        study.optimize(partial(train, conf=conf, dataset=dataset), n_trials=n_trials)
    else:
        print(f"Rank {rank}", flush=True)
        for _ in range(n_trials):
            try:
                train(None, conf, dataset)
            except optuna.TrialPruned:
                pass

    if rank == 0:
        best_parameters = study.best_params
        print(best_parameters)