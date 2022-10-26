import os
from os import path
import yaml
import sys
import torch

from datasets import SWATDataset
from training import Trainer, hyperparameter_optimize
from evaluation import NNEvaluator
from invariants import generate_predicates, InvariantMiner

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

    if task == "train":
        dataset = SWATDataset(conf, conf["data"]["normal"],
                              sequence_len=conf["model"]["sequence_length"],
                              train=True,
                              load_scaler=False)
        trainer = Trainer(conf, dataset)
        trainer.train_prediction()
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
