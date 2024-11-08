import copy
import os
from sched import scheduler
import sys
from os import path
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
from ray.tune.tune_config import TuneConfig
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.search.hyperopt import HyperOptSearch

from datasets import GridDataset, GridGraphDataset
from models import GCN
from training import LocalizationTrainer
from equations import build_equations
from utils import generate_fdia


def train_localize(config: dict=None):
    if config is not None:
        conf["model"]["n_heads"] = config.get("n_heads", conf["model"]["n_heads"])
        conf["model"]["hidden_size"] = config.get("size", conf["model"]["hidden_size"])
        conf["model"]["n_layers"] = config.get("n_layers", conf["model"]["n_layers"])
        conf["model"]["n_iters"] = config.get("n_iters", conf["model"]["n_iters"])
        conf["model"]["n_stacks"] = config.get("n_stacks", conf["model"]["n_stacks"])
        conf["model"]["k"] = config.get("k", conf["model"]["k"])
        conf["train"]["lr"] = config.get("lr", conf["train"]["lr"])
        conf["train"]["regularization"] = config["regularization"]
        conf["model"]["noralization"] = config.get("normalization", conf["model"]["noralization"])
    # pinn_model = load_pinn()
    if use_graph:
        dataset = GridGraphDataset(conf, conf["data"]["attack"])
    else:
        dataset = GridDataset(conf, conf["data"]["attack"])
    
    if fraction > 0:
        idx = range(int(fraction * len(dataset)))
        dataset  = Subset(dataset, idx)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_fraction, validate_fraction])
    trainer = LocalizationTrainer(conf)
    if ray.is_initialized():
        trainer.train(0, train_dataset, val_dataset)
    else:
        mp.spawn(trainer.train, args=(train_dataset, val_dataset),
                 nprocs=gpus,
                 join=True)


def localize():
    if use_graph:
        dataset = GridGraphDataset(conf, conf["data"]["test"], False)
    else:
        dataset = GridDataset(conf, conf["data"]["test"])
    trainer = LocalizationTrainer(conf)
    mp.spawn(trainer.localize, args=(dataset,),
             nprocs=gpus,
             join=True)
    
def hyperparameter_optimize():
    # conf["data"]["normal"] = path.abspath(conf["data"]["normal"])
    conf["data"]["attack"] = path.abspath(conf["data"]["attack"])
    conf["data"]["ybus"] = path.abspath(conf["data"]["ybus"])
    conf["train"]["checkpoint_dir"] = path.abspath(conf["train"]["checkpoint_dir"])
    conf["train"]["tune_checkpoint"] = path.abspath(conf["train"]["tune_checkpoint"])
    population_training = conf["train"].get("population_train", 1)
    config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "dropout": tune.uniform(0.1, 0.8),
        "regularization": tune.loguniform(1e-7, 1e-3)
    }
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
    ray.init(_temp_dir=path.abspath("../ray"))

    algo = HyperOptSearch(metric="loss", mode="min")

    if not population_training:
        config.update({
            "n_stacks": tune.choice([2, 3, 4]),
            "k": tune.choice([2, 3, 4]),
            # "n_heads": tune.choice([i for i in range(2, 9)]),
            "size": tune.choice([2 ** i for i in range(4, 11)]),
            "n_layers": tune.choice([i for i in range(2, 8)]),
            "n_iters": tune.choice([2, 3, 4, 5]),
            "normalization": tune.choice(["sym", "rw"])
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
    
    tunable = train_localize
    trainer = TorchTrainer(
        tunable,
        scaling_config=ScalingConfig(num_workers=1, use_gpu=conf["train"]["cuda"]))
    
    if conf["train"]["load_checkpoint"]:
        tuner = tune.Tuner.restore(
            conf["train"]["tune_checkpoint"],
            trainer,
            restart_errored=True
        )
    else:
        tuner = tune.Tuner(
        trainer,
        param_space={
            "train_loop_config": config
        },
        tune_config=TuneConfig(
            num_samples=conf["train"]["num_samples"],
            scheduler=scheduler,
            search_alg=algo,
            max_concurrent_trials=4
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

def load_pinn():
    pinn_ckpt = torch.load(pinn_ckpt_path)
    config = copy.deepcopy(conf)
    config["model"] = pinn_ckpt["structure"]
    pinn_model = GCN(config, 2, 2)
    pinn_model.load_state_dict(pinn_ckpt["model"])
    return pinn_model

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
    pinn_ckpt_path = path.join(conf["train"]["checkpoint_dir"], conf["train"]["checkpoint"], "pinn.pt")
    train_fraction = conf["train"]["train_fraction"]
    validate_fraction = conf["train"]["validate_fraction"]
    find_error_fraction = conf["train"]["find_error_fraction"]
    fraction = conf["train"]["fraction"]
    gpus = conf["train"]["gpus"]
    use_graph = conf["model"]["graph"]

    if task == "train_localize":
        train_localize()
    elif task == "localize":
        localize()
    elif task == "hyperparameter_optimize":
        hyperparameter_optimize()
    elif task == "equ_error":
        losses = []
        if use_graph:
            dataset = GridGraphDataset(conf, conf["data"]["attack"])
            equations = build_equations(conf)
            loader = DataLoader(dataset, batch_size=conf["train"]["batch_size"])
            for batch in loader:
                for equ in equations:
                    loss = equ.evaluate(batch)
                    losses.append(loss)
            loss = torch.cat(losses)
            print("average loss:", torch.mean(loss))
            print("Standard dev:", torch.std(loss))

    elif task == "attack_generation":
        generate_fdia(conf)
    else:
        raise RuntimeError("Unknown task")