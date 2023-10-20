import  os
from os import path
import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import FCN
from equations import build_equations

class PINNTrainer:

    def __init__(self, conf, dataset) -> None:
        self.conf = conf
        self.batch_size = conf["train"]["batch_size"]
        self.scale = conf["train"]["scale"]

        self.n_buses = conf["data"]["n_buses"]
        self.checkpoint_dir = path.join(conf["train"]["checkpoint_dir"], conf["train"]["checkpoint"])
        self.checkpoint_path = path.join(self.checkpoint_dir, "model.pt")
        self.size = self.conf["train"]["gpus"]

        self.model = FCN(conf, 4 * self.n_buses, 2 * self.n_buses)
        self.equations = build_equations(conf, dataset.get_categorical_features(), 
                                         dataset.get_continuous_features())
        
    def _init_ddp(self, rank, datasets, shuffles, backend="nccl"):
        device = torch.device(f"cuda:{rank}")
        # self.model = self.model.to(device)
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        
        dist.init_process_group(backend, rank=rank, world_size=self.size)

        self.model = torch.nn.parallel.DistributedDataParallel(self.model) #, device_ids=[rank], output_device=rank)

        loaders = []
        for dataset, shuffle in zip(datasets, shuffles):
            sampler = DistributedSampler(dataset,
                                         num_replicas=self.size,
                                         rank=rank,
                                         shuffle=shuffle)
            loader = DataLoader(dataset,
                                batch_size=self.batch_size,
                                shuffle=False,
                                sampler=sampler)
            loaders.append(loader)
        return loaders

    def train(self, rank, train_dataset, val_dataset):
        start_epoch = 0
        best_loss = torch.inf
        optim = torch.optim.Adam(self.model.parameters(), lr = self.conf["train"]["lr"], weight_decay=self.conf["train"]["regularization"])

        if self.conf["train"]["load_checkpoint"] and path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path)
            start_epoch = checkpoint["epoch"]
            self.model.load_state_dict(checkpoint["model"])
            optim.load_state_dict(checkpoint["optim"])
            best_loss = checkpoint["loss"]

        if rank == 0:
            print(self.model)
        train_loader, val_loader = self._init_ddp(rank, [train_dataset, val_dataset], [True, False], "gloo")
        epochs = self.conf["train"]["epochs"]
        if rank == 0:
            tb_writer = SummaryWriter(self.checkpoint_dir)

        for epoch in range(start_epoch, epochs):
            if rank == 0:
                loader = tqdm(train_loader, position=0, leave=True)
            else:
                loader = train_loader
            for inputs, targets in loader:
                optim.zero_grad()
                data_loss, physics_loss, loss = self._combine_losses(inputs, targets)
                loss.backward()
                optim.step()

                if rank == 0:
                    loader.set_description(f"Train Epoch [{epoch: 3d}/{epochs}]")
                    loader.set_postfix(loss=loss.item(), data_loss=data_loss.item(), physics_loss=physics_loss.item())
            
            if rank == 0:
                loader = tqdm(val_loader)
            else:
                loader = val_loader

            with torch.no_grad():
                total_data_loss = 0
                total_physics_loss = 0
                total_loss = 0
                for inputs, targets in loader:
                    data_loss, physics_loss, loss = self._combine_losses(inputs, targets)
                    total_data_loss += data_loss
                    total_physics_loss += physics_loss
                    total_loss += loss

                    if rank == 0:
                        loader.set_description(f"Val Epoch [{epoch: 3d}/{epochs}]")
                        loader.set_postfix(loss=loss.item(), data_loss=data_loss.item(), physics_loss=physics_loss.item())
                total_data_loss /= len(val_loader)
                total_physics_loss /= len(val_loader)
                total_loss /= len(val_loader)

                dist.all_reduce(total_data_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_physics_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)

                if rank == 0:
                    total_data_loss /= self.size
                    total_physics_loss /= self.size
                    total_loss /= self.size

                    tb_writer.add_scalar("Data loss", total_data_loss, epoch)
                    tb_writer.add_scalar("Physics loss", total_physics_loss, epoch)
                    tb_writer.add_scalar("Loss", total_loss, epoch)

                    if loss < best_loss:
                        checkpoint = {
                            "model": self.model.module.state_dict(),
                            "optim": optim.state_dict(),
                            "loss": loss.item(),
                            "epoch": epoch
                        }
                        torch.save(checkpoint, self.checkpoint_path)
            dist.barrier()

    def create_normal_profile(self, rank, dataset):
        if not path.exists(self.checkpoint_path):
            if rank == 0:
                print("Checpoint does not exist")
            return
        checkpoint = torch.load(self.checkpoint_path)
        self.model.load_state_dict(checkpoint["model"])
        loader = self._init_ddp(rank, [dataset], [False], "gloo")[0]

        losses = []
        if rank == 0:
            data_loader = tqdm(loader)
        else:
            data_loader = loader

        for inputs, targets in data_loader:
            data_loss, physics_loss = self._compute_loss(inputs, targets)
            data_loss = data_loss.mean(dim=1).view(-1, 1)
            loss = torch.cat([data_loss, physics_loss], dim=1)
            losses.append(loss)
        loss = torch.cat(losses)

        if rank == 0:
            all_losses = [torch.empty(loss.size()) for _ in range(self.size)]
            dist.gather(loss, all_losses, 0)
            losses = torch.cat(all_losses)
            
            mean = torch.mean(losses, dim=0)
            std = torch.std(losses, dim=0)
            checkpoint["mean"] = mean
            checkpoint["std"] = std
            torch.save(checkpoint, self.checkpoint_path)
        else:
            dist.gather(loss, [], 0)
        dist.barrier()

    def _combine_losses(self, inputs, targets):
        data_loss, physics_loss = self._compute_loss(inputs, targets)
        data_loss = data_loss.mean()
        physics_loss = physics_loss.mean()
        loss = self.scale[0] * data_loss + self.scale[1] * physics_loss
        return data_loss, physics_loss, loss

    def _compute_loss(self, inputs, targets):
        predicted = self.model(inputs)
        data_loss = F.mse_loss(predicted, targets, reduction='none')

        physics_loss = torch.zeros((len(inputs), len(self.equations)))
        for i, equation in enumerate(self.equations):
            physics_loss[:, i] = equation.confidence_loss(inputs, predicted, targets)

        return data_loss, physics_loss