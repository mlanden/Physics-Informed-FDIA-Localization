import os
from os import path
import yaml
import sys
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from torch.utils.data import Subset, DataLoader
import torch.multiprocessing as mp

from pytorch_lightning.loggers import TensorBoardLogger
from ray import air, tune
from ray.air import session
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback

from datasets import SWATDataset
from training import ICSTrainer
from utils import make_roc_curve
from invariants import generate_predicates, InvariantMiner


def train(config=None):
    callbacks = [RichProgressBar(leave=True)]
    if config is not None:
        print(os.getcwd())
        callbacks.append(TuneReportCheckpointCallback(
            metrics={
                "loss": "val_loss"
            },
            filename="checkpoint",
            on="validation_end"
        ))
        conf["train"]["lr"] = config["lr"]
        conf["train"]["batch_size"] = config["batch_size"]
    else:
        callbacks.append(ModelCheckpoint(dirpath=checkpoint_dir,
                                         filename=checkpoint,
                                         save_last=True,
                                         every_n_train_steps=0,
                                         every_n_epochs=1,
                                         save_on_train_epoch_end=True)
                         )

    trainer = Trainer(default_root_dir=checkpoint_dir,
                      log_every_n_steps=10,
                      max_epochs=conf["train"]["epochs"],
                      devices=1,
                      accelerator="gpu" if torch.cuda.is_available() else "cpu",
                      callbacks=callbacks
                      )
    dataset = SWATDataset(conf, conf["data"]["normal"],
                          window_size=conf["model"]["window_size"],
                          train=True,
                          load_scaler=False)
    datalen = len(dataset)
    train_len = int(datalen * train_fraction)
    print("train frac", train_len)
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
    trainer = Trainer(default_root_dir=checkpoint_dir,
                      # devices=1,
                      # accelerator="gpu" if torch.cuda.is_available() else "cpu",
                      )

    dataset = SWATDataset(conf, conf["data"]["normal"],
                          window_size=1,
                          train=True,
                          load_scaler=True)
    model = ICSTrainer.load_from_checkpoint(checkpoint_to_load, conf=conf)
    start = int((train_fraction + validate_fraction) * len(dataset))
    size = int(find_error_fraction * len(dataset))
    idx = list(range(start, start + size))
    normal = Subset(dataset, idx)
    print(f"States to test:", len(normal))
    loader = DataLoader(normal, batch_size=batch_size)

    trainer.test(model, loader)


def test():
    trainer = Trainer(default_root_dir=checkpoint_dir,
                      devices=1,
                      accelerator="gpu" if torch.cuda.is_available() else "cpu",
                      )
    dataset = SWATDataset(conf, conf["data"]["attack"],
                          window_size=1,
                          train=False,
                          load_scaler=True)
    type_ = conf["train"]["type"]
    if type_ == "prediction":
        model = ICSTrainer.load_from_checkpoint(checkpoint_to_load, conf=conf)
        loader = DataLoader(dataset, batch_size=batch_size, drop_last=False)
        trainer.predict(model, loader)
    elif type_ == "invariants":
        miner = InvariantMiner(conf, dataset)
        miner.evaluate()
    else:
        raise RuntimeError("Unknown evaluation type")


def hyperparameter_optimize():
    conf["data"]["normal"] = path.abspath(conf["data"]["normal"])
    conf["train"]["checkpoint_dir"] = path.abspath(conf["train"]["checkpoint_dir"])

    config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128])
    }

    scheduler = ASHAScheduler(max_t=conf["train"]["epochs"],
                              grace_period=1,
                              reduction_factor=2)
    reporter = CLIReporter(parameter_columns=["lr", "batch_size"],
                           metric_columns=["loss", "training_iterations"])

    resources = {"cpu": 1, "gpu": 0.5}

    tuner = tune.Tuner(
        tune.with_resources(train,
                            resources=resources),
        tune_config=tune.TuneConfig(metric="loss",
                                    mode="min",
                                    scheduler=scheduler,
                                    num_samples=conf["train"]["num_samples"]),
        run_config=air.RunConfig(name="tune_ics",
                                 progress_reporter=reporter),
        param_space=config
    )
    results = tuner.fit()
    print("Best hyperparameters:", results.get_best_result().config)


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
    checkpoint_to_load = path.join(checkpoint_dir, "last.ckpt")
    results_dir = path.join("results", conf["train"]["checkpoint"])
    if not path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    if not path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
    mp.set_start_method("spawn", force=True)
    mp.set_sharing_strategy("file_system")

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
        train_fraction = 0.01
        hyperparameter_optimize()
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
    elif task == "invariants":
        dataset = SWATDataset(conf, conf["data"]["normal"],
                              sequence_len=conf["model"]["sequence_length"],
                              train=True,
                              load_scaler=False)
        miner = InvariantMiner(conf, dataset)
        miner.mine_invariants()
    elif task == "roc":
        losses_path = path.join(checkpoint_dir, "evaluation_losses.json")
        make_roc_curve(losses_path)
    else:
        raise RuntimeError(f"Unknown task: {task}")
