import os
from os import path
import yaml
import sys

from datasets import SWATDataset
from training import rnn_train, find_normal_error


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: main.py config")
        quit(1)

    conf_path = sys.argv[1]
    with open(conf_path, "r") as fd:
        conf = yaml.safe_load(fd)
    checkpoint_dir = path.join("checkpoint", conf["train"]["checkpoint"])
    if not path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok = True)

    task = conf["task"]
    if task == "train":
        dataset = SWATDataset(conf, conf["data"]["normal"],
                              sequence_len = conf["model"]["sequence_length"],
                              train = True,
                              load_scaler = False)
        rnn_train(conf, dataset)
    elif task == "threshold":
        dataset = SWATDataset(conf, conf["data"]["normal"],
                              sequence_len = 1,
                              train = True,
                              load_scaler = True)
        find_normal_error(conf, dataset)