import torch.nn as nn


activations = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "leakyRelu": nn.LeakyReLU()
}