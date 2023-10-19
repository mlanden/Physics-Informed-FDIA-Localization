import  os
import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from tqdm import tqdm

from models import FCN
from equations import build_equations

class PINNTrainer:

    def __init__(self, conf, dataset) -> None:
        self.conf = conf
        self.batch_size = conf["train"]["batch_size"]
        self.scale = conf["train"]["scale"]

        self.n_buses = conf["data"]["n_buses"]
        self.model = FCN(conf, 4 * self.n_buses, 2 * self.n_buses)
        self.equations = build_equations(conf, dataset.get_categorical_features(), 
                                         dataset.get_continuous_features())

    def _init_ddp(self, rank, datasets, shuffles, backend="nccl"):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        size = self.conf["train"]["gpus"]
        dist.init_process_group(backend, rank=rank, world_size=size)

        self.model = torch.nn.parallel.DistributedDataParallel(self.model) #, device_ids=[rank], output_device=rank)

        loaders = []
        for dataset, shuffle in zip(datasets, shuffles):
            sampler = DistributedSampler(dataset,
                                         num_replicas=size,
                                         rank=rank,
                                         shuffle=shuffle)
            loader = DataLoader(dataset,
                                batch_size=self.batch_size,
                                shuffle=False,
                                sampler=sampler)
            loaders.append(loader)
        return loaders

    def train(self, rank, train_dataset, val_dataset):
        device = torch.device(f"cuda:{rank}")
        # self.model = self.model.to(device)
        train_loader, val_loader = self._init_ddp(rank, [train_dataset, val_dataset], [True, False], "gloo")
        epochs = self.conf["train"]["epochs"]
        print(f"Rank: {dist.get_rank()}")

        mse_loss = torch.nn.MSELoss(reduce="none")
        optim = torch.optim.Adam(self.model.parameters(), lr = self.conf["train"]["lr"])
        for epoch in range(epochs):
            loader = tqdm(train_loader)
            for inputs, targets in loader:
                optim.zero_grad()
                predicted = self.model(inputs)
                data_loss = mse_loss(predicted, targets)

                physics_loss = torch.zeros((len(inputs), len(self.equations)))
                for i, equation in enumerate(self.equations):
                    physics_loss[:, i] = equation.confidence_loss(inputs, predicted, targets)
                physics_loss = torch.sum(physics_loss, dim=1)

                loss = self.scale[0] * data_loss + self.scale[1] * physics_loss
                loss = loss.mean()
                loss.backward()
                optim.step()


                loader.set_description(f"Epoch [{epoch: 3d}/{epochs}]")
                loader.set_postfix(loss=loss.item(), data_loss=data_loss.mean().item(), physics_loss=physics_loss.mean().item())
        