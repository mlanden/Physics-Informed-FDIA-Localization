import os.path
from functools import partial

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch

from .trainer import Trainer


def train(config, conf, dataset, device):
    conf["train"]["lr"] = config["lr"]
    conf["train"]["batch_size"] = config["batch_size"]

    trainer = Trainer(conf, dataset, device)
    trainer.train_prediction()


def hyperparameter_optimize(conf: dict, dataset, device):
    space = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([2 ** i for i in range(4, 8)])
    }
    # search_alg = HyperOptSearch()
    scheduler = ASHAScheduler()
    reporter = CLIReporter(
        metric_columns = ["loss"],
        print_intermediate_tables = True,
        max_report_frequency = 60,
        )
    result = tune.run(
        partial(train, conf=conf, dataset=dataset, device=device),
        resources_per_trial={"cpu": conf["train"]["n_workers"]},
        num_samples=2,
        scheduler=scheduler,
        config=space,
        # search_alg=search_alg,
        metric="loss",
        mode="min",
        max_concurrent_trials=2,
        # progress_reporter=reporter,
    )
    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))