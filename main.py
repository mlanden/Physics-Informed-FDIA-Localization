import os
from os import path
from pprint import pprint
import yaml
import sys
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar, LearningRateMonitor, EarlyStopping
from pytorch_lightning.plugins import DDPPlugin
from torch.utils.data import Subset, DataLoader
import torch.multiprocessing as mp
import numpy as np

from datasets import SWATDataset, GridDataset
from training import ICSTrainer, EquationDetector
from analysis import investigate_invariants
from invariants import generate_predicates, InvariantMiner
from equations import build_equations


def get_dataset(conf, data_path, train, load_scalar, window_size):
    type = conf["data"]["type"]
    if type == "swat":
        return SWATDataset(conf, data_path, window_size, train, load_scalar)
    elif type == "grid":
        return GridDataset(conf, data_path, window_size, train, load_scalar)


def train(config=None):
    callbacks = [RichProgressBar(leave=True), LearningRateMonitor("epoch"),
                 EarlyStopping(monitor="val_loss", patience=15,)
                 ]
    if config is not None:
        from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
        callbacks.append(TuneReportCheckpointCallback(
            metrics={
                "loss": "unscaled_val_loss"
            },
            filename="checkpoint",
            on="validation_end"
        ))
        # Prediction
        # conf["model"]["hidden_size"] = config["hidden_size"]
        # conf["model"]["n_layers"] = config["n_layers"]
        if "regularization" in config:
            conf["train"]["regularization"] = config["regularization"]

        # Autoencoder
        if "initial_size" in config:
            conf["model"]["layer_sizes"] = [config["initial_size"]]
            for i in range(config["n_layers"]):
                conf["model"]["layer_sizes"].append(conf["model"]["layer_sizes"][-1] // 2)
        
        if "equ_scale" in config:
            conf["train"]["scale"][1] = config["equ_scale"]
    else:
        callbacks.append(ModelCheckpoint(dirpath=checkpoint_dir,
                                         filename=checkpoint,
                                         save_last=True,
                                         # every_n_train_steps=10,
                                         # every_n_epochs=1,
                                         save_on_train_epoch_end=True,
                                         save_top_k=1,
                                         verbose=True)
                         )
    num_nodes = conf["train"]["nodes"]
    batch_size = conf["train"]["batch_size"]
    batch_size = batch_size // gpus // num_nodes
    trainer = Trainer(default_root_dir=checkpoint_dir,
                      max_epochs=conf["train"]["epochs"],
                      devices=gpus,
                      num_nodes=num_nodes,
                      strategy=DDPPlugin(find_unused_parameters=False),
                      accelerator="gpu" if torch.cuda.is_available() else "cpu",
                      callbacks=callbacks,
                    #   limit_train_batches=3
                    #   track_grad_norm=2,
                      gradient_clip_val=0.5
                      )
    dataset = get_dataset(conf,conf["data"]["normal"], True, False, conf["model"]["window_size"])
    datalen = len(dataset)
    invariant_len = int(datalen * invariant_fraction)
    train_len = int(datalen * train_fraction)
    train_idx = list(range(invariant_len, train_len + invariant_len))
    train_data = Subset(dataset, train_idx)
    val_len = int(datalen * validate_fraction)
    val_idx = list(range(invariant_len + train_len, invariant_len + train_len + val_len))
    validation_data = Subset(dataset, val_idx)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False, drop_last=False)
    torch.autograd.set_detect_anomaly(True)
    
    model = ICSTrainer(conf, dataset.get_categorical_features(), dataset.get_continuous_features())
    if load_checkpoint and path.exists(checkpoint_to_load):
        trainer.fit(model, train_loader, val_loader,
                    ckpt_path=checkpoint_to_load)
    else:
        if load_checkpoint:
            raise RuntimeError("Could not find checkpoint")
        trainer.fit(model, train_loader, val_loader)


def find_normal_error():
    trainer = Trainer(default_root_dir=checkpoint_dir,
                      devices=gpus,
                      accelerator="gpu" if torch.cuda.is_available() else "cpu",
                      )

    dataset = get_dataset(conf, conf["data"]["normal"], True, True, conf["model"]["window_size"])
    
    model = ICSTrainer.load_from_checkpoint(checkpoint_to_load, conf=conf)
    print("Testing with", checkpoint_to_load)
    start = int((invariant_fraction + train_fraction + validate_fraction) * len(dataset))
    size = int(find_error_fraction * len(dataset))
    idx = list(range(start, start + size))
    normal = Subset(dataset, idx)
    print(f"States to test:", len(normal))
    loader = DataLoader(normal, batch_size=batch_size)

    trainer.test(model, loader)


def test():
    dataset = get_dataset(conf, conf["data"]["attack"], False, True, 1)
    idx = list(range(1000))
    # dataset = Subset(dataset, idx)
    type_ = conf["train"]["type"]
    if type_ == "prediction":
        trainer = Trainer(default_root_dir=checkpoint_dir,
                          devices=gpus,
                          accelerator="gpu" if torch.cuda.is_available() else "cpu",
                          )
        model = ICSTrainer.load_from_checkpoint(checkpoint_to_load, conf=conf)
        loader = DataLoader(dataset, batch_size=batch_size, drop_last=False)
        trainer.predict(model, loader)
    elif type_ == "invariants":
        miner = InvariantMiner(conf, dataset)
        miner.evaluate()
    else:
        raise RuntimeError("Unknown evaluation type")


def hyperparameter_optimize():
    from ray import air, tune
    from ray.tune import CLIReporter
    from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining

    conf["data"]["normal"] = path.abspath(conf["data"]["normal"])
    conf["train"]["checkpoint_dir"] = path.abspath(conf["train"]["checkpoint_dir"])
    conf["train"]["epocs"] = 25
    global gpus
    gpus = 1

    config = {
        # "regularization": tune.loguniform(1e-4, 1),
        # "hidden_size": tune.choice([10 * i for i in range(1, 11)]),
        # "initial_size": tune.choice([2 ** i for i in range(4, 10)]),
        # "n_layers": tune.choice([1, 2, 3, 4]),
        "equ_scale": tune.loguniform(1e-3, 1)
    }

    scheduler = ASHAScheduler(max_t=conf["train"]["epochs"],
                              grace_period=15,
                              reduction_factor=2)
    reporter = CLIReporter(parameter_columns=list(config.keys()),
                           metric_columns=["loss", "training_iterations"])

    resources = {"gpu": 1}
    tuner = tune.Tuner(
        tune.with_resources(train,
                            resources=resources),
        tune_config=tune.TuneConfig(metric="loss",
                                    mode="min",
                                    scheduler=scheduler,
                                    num_samples=conf["train"]["num_samples"]),
        run_config=air.RunConfig(name="tune_ics",
                                 progress_reporter=reporter),
        param_space=config,

    )
    results = tuner.fit()
    print("Best hyperparameters:", results.get_best_result())
    print(results.get_best_result().config)


def equation_detect():
    dataset = get_dataset(conf,conf["data"]["normal"], True, False, conf["model"]["window_size"])
    detector = EquationDetector(conf, dataset.get_categorical_features(), dataset.get_continuous_features())

    for batch in dataset:
        detector.training_step(batch)

    dataset = get_dataset(conf, conf["data"]["attack"], False, True, 1)
    attack_map = dataset.get_attack_map()
    for i, state in enumerate(dataset):
        attack = -1
        for idx in attack_map:
            if i in attack_map[idx]:
                attack = idx
        if attack == 6:
            print(i, end=" ")
        detector.detect(state, attack)
    detector.print_stats(dataset.get_attack_map())

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: main.py config")
        quit(1)

    conf_path = sys.argv[1]
    with open(conf_path, "r") as fd:
        conf = yaml.safe_load(fd)
    checkpoint = conf["train"]["checkpoint"]

    checkpoint_dir = path.join("checkpoint", checkpoint)
    checkpoint_to_load = path.join(checkpoint_dir, f"last.ckpt")  # s
    results_dir = path.join("results", conf["train"]["checkpoint"])
    if not path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    if not path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
    mp.set_start_method("spawn", force=True)
    mp.set_sharing_strategy("file_system")

    task = conf["task"]
    print("Task:", task)
    gpus = conf["train"]["gpus"]
    invariant_fraction = conf["train"]["invariant_fraction"]
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
                              window_size=1,
                              train=True,
                              load_scaler=False)
        predicates = generate_predicates(dataset, conf)
    elif task == "invariants":
        dataset = SWATDataset(conf, conf["data"]["normal"],
                              window_size=1,
                              train=True,
                              load_scaler=False)
        miner = InvariantMiner(conf, dataset)
        miner.mine_invariants()
    elif task == "analyze":
        dataset = SWATDataset(conf, conf["data"]["attack"],
                              window_size=1,
                              train=False,
                              load_scaler=True)
        type_ = conf["train"]["type"]
        trainer = Trainer(default_root_dir=checkpoint_dir,
                          devices=gpus,
                          accelerator="gpu" if torch.cuda.is_available() else "cpu",
                          )
        model = ICSTrainer.load_from_checkpoint(checkpoint_to_load, conf=conf)
        investigate_invariants(conf, checkpoint_dir, dataset, model)
    elif task == "equations":
        equation_detect()
    elif task == "equ_error":
        dataset = get_dataset(conf,conf["data"]["normal"], True, False, conf["model"]["window_size"])
        equations = build_equations(conf, dataset.get_categorical_features(), dataset.get_continuous_features())
        features, labels = dataset.get_data()
        losses = []
        for i in range(len(features)):
            loss = equations[0].evaluate(features[i, :])
            # break
            losses.append(loss)
        print("Average loss:", np.mean(losses))
        equations[0].loss_plot()

    else:
        raise RuntimeError(f"Unknown task: {task}")
