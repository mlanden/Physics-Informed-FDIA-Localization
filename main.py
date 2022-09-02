import torch.nn
import yaml
import sys

from datasets import SWATDataset
from training import rnn_train


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: main.py config")
        quit(1)
    conf_path = sys.argv[1]
    with open(conf_path, "r") as fd:
        conf = yaml.safe_load(fd)

    data = SWATDataset(conf["data"]["normal"],
                       sequence_len = conf["model"]["sequence_length"], window_size = conf["data"]["window_size"],
                       read_normal = False)
    rnn_train(conf, data)