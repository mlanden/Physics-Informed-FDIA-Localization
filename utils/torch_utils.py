import os
import torch.nn as nn
import torch.distributed as dist


activations = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "leakyRelu": nn.LeakyReLU()
}


def launch_distributed(rank, size, function):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(dist.Backend.GLOO, rank = rank, world_size = size)
    function()