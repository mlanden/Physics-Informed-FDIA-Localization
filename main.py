import torch.nn
import yaml
import sys

from torch.utils.data import DataLoader
from torch import optim
from datasets import SWATDataset
from models import PredictionModel

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: main.py config")
        quit(1)
    conf_path = sys.argv[1]
    with open(conf_path, "r") as fd:
        conf = yaml.safe_load(fd)

    data = SWATDataset(conf["data"]["normal"],
                       sequence_len = conf["model"]["sequence_length"], window_size = conf["data"]["window_size"])
    normal_data = DataLoader(data, batch_size = conf["train"]["batch_size"], shuffle = True)

    model = PredictionModel(conf)
    optimizer = optim.Adam(model.parameters(), lr = conf["train"]["lr"])
    loss_fn = torch.nn.MSELoss()
    epochs = conf["train"]["epochs"]
    for i in range(epochs):
        for seq, target in normal_data:
            predicted = model(seq)
            loss = loss_fn(predicted, target)

            optimizer.zero_grad()
            loss.backeard()
            optimizer.step()
