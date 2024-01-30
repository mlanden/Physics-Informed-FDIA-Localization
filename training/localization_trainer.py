import  os
import tempfile
from os import path
import ray
from ray import train
from sklearn.metrics import confusion_matrix, roc_curve
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch_geometric.loader import DataLoader as gDataLoader
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

from models import GCN
from equations import build_equations

class LocalizationTrainer:

    def __init__(self, conf):
        self.conf = conf
        self.batch_size = conf["train"]["batch_size"]
        self.checkpoint_dir = path.join(conf["train"]["checkpoint_dir"], conf["train"]["checkpoint"])
        self.pinn_model_path = path.join(self.checkpoint_dir, "pinn.pt") 
        self.localize_model_path = path.join(self.checkpoint_dir, "localize.pt")
        self.size = self.conf["train"]["gpus"]

        self.pinn_model = GCN(self.conf, 2, 2)
        self.localize_model = GCN(self.conf, 2, self.conf["model"]["hidden_size"], dense=True)

    def _init_ddp(self, rank, datasets, shuffles):
        if not ray.is_initialized():
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = '29500'
            if self.conf["train"]["cuda"]:
                backend = "nccl"
            else:
                backend = "gloo"
            dist.init_process_group(backend, rank=rank, world_size=self.size)
            if backend == "gloo":
                self.model = torch.nn.parallel.DistributedDataParallel(self.localize_model)
            else:
                self.model = torch.nn.parallel.DistributedDataParallel(self.localize_model,
                                                                        device_ids=[rank],
                                                                        output_device=rank)
        else:
            if not train.get_checkpoint():
                self.model = train.torch.prepare_model(self.model)

        loaders = []
        for dataset, shuffle in zip(datasets, shuffles):
            sampler = DistributedSampler(dataset,
                                         num_replicas=self.size,
                                         rank=rank,
                                         shuffle=shuffle)
            loader = gDataLoader(dataset,
                                batch_size=self.batch_size,
                                shuffle=False,
                                sampler=sampler)
            loaders.append(loader)
        return loaders
    
    def train(self, rank, train_dataset, val_dataset):
        start_epoch = 0
        best_loss = torch.inf
        if not path.exists(self.pinn_model_path):
            if rank == 0:
                print("PINN model does not exist")
            return
        
        self.pinn_model_ckpt = torch.load(self.pinn_model_path)
        self.pinn_model.load_state_dict(self.pinn_model_ckpt["model"])
        self.pinn_model.eval()

        optim = torch.optim.Adam(self.localize_model.parameters(), 
                                 lr=self.conf["train"]["lr"],
                                 weight_decay=self.conf["train"]["regularization"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.5)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        
        device = torch.device(f"cuda:{rank}") if self.conf["train"]["cuda"] else torch.device("cpu")
        self.pinn_model = self.pinn_model.to(device)
        self.localize_model = self.localize_model.to(device)

        checkpoint = None
        if ray.ray.is_initialized():
            checkpoint = train.get_checkpoint()
            if checkpoint:
                with checkpoint.as_directory() as checkpoint_dir:
                    checkpoint = torch.load(path.join(checkpoint_dir, self.model_name))
        elif self.conf["train"]["load_checkpoint"] and path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path)
        
        if checkpoint is not None:
            start_epoch = checkpoint["epoch"]
            self.model.load_state_dict(checkpoint["model"])
            optim.load_state_dict(checkpoint["optim"])
            scheduler.load_state_dict(checkpoint["schedule"])
            best_loss = checkpoint["loss"]

        train_loader, val_loader = self._init_ddp(rank, [train_dataset, val_dataset], [True, False])
        rank = dist.get_rank()
        if rank == 0:
            print(self.model)
        epochs = self.conf["train"]["epochs"]

        if rank == 0 and not ray.is_initialized():
            tb_dir = path.join(self.checkpoint_dir, "localization_training")
            if not path.exists(tb_dir):
                os.makedirs(tb_dir)
            version = len([name for name in os.listdir(tb_dir) if path.isdir(path.join(tb_dir, name))]) + 1
            print("Version:", version)
            tb_writer = SummaryWriter(path.join(tb_dir, f"version_{version}"))

        for epoch in range(start_epoch, epochs):
            self.localize_model.train()
            train_loader.sampler.set_epoch(epoch)
            if rank == 0 and not ray.is_initialized():
                loader = tqdm(train_loader, position=0, leave=True)
            else:
                loader = train_loader
            total_loss = 0
            for data in loader:
                optim.zero_grad()
                data = data.to(device)

                embedding = self.pinn_model(data)
                data.x = embedding
                logits = self.localize_model(data)
                loss = loss_fn(logits, data.classes)
                loss.backward()
                optim.step()
                total_loss += loss.detach()

                if rank == 0 and not ray.is_initialized():
                    loader.set_description(f"Train Epoch [{epoch: 3d}/{epochs}]")
                    loader.set_postfix(loss=loss.item())
            total_loss /= len(train_loader)
            dist.all_reduce(total_loss)
            total_loss /= self.size
            if rank == 0 and not ray.is_initialized():
                tb_writer.add_scalar("Training loss", total_loss, epoch)

            val_loader.sampler.set_epoch(epoch)
            if rank == 0 and not ray.is_initialized():
                loader = tqdm(val_loader)
            else:
                loader = val_loader

            self.localize_model.eval()
            with torch.no_grad():
                total_loss = 0
                for data in loader:
                    data = data.to(device)
                    embedding = self.pinn_model(data)
                    data.x = embedding
                    logits = self.localize_model(data)
                    loss = loss_fn(logits, data.classes)
                    total_loss += loss

                    if rank == 0 and not ray.is_initialized():
                        loader.set_description(f"Val Epoch [{epoch: 3d}/{epochs}]")
                        loader.set_postfix(loss=loss.item())

                total_loss /= len(val_loader)
                dist.all_reduce(total_loss)
                total_loss /= self.size
                
                scheduler.step(total_loss)
                
                if rank == 0:
                    lr = optim.param_groups[0]['lr']
                    if not ray.is_initialized():
                        tb_writer.add_scalar("Learning Rate", lr, epoch)
                        tb_writer.add_scalar("Validation Loss", total_loss, epoch)

                    if loss < best_loss:
                        try:
                            model = self.model.module
                        except AttributeError:
                            model = self.model
                        checkpoint = {
                            "model": model.state_dict(),
                            "optim": optim.state_dict(),
                            "schedule": scheduler.state_dict(),
                            "loss": total_loss.item(),
                            "epoch": epoch
                        }
                        if ray.is_initialized():
                            with tempfile.TemporaryDirectory() as tempdir:
                                torch.save(checkpoint, path.join(tempdir, self.model_name))
                                train.report({"loss": total_loss.item()}, checkpoint=
                                            train.Checkpoint.from_directory(tempdir))
                                print(f"Epoch {epoch}: Loss: {total_loss.item()}")
                        else:
                            torch.save(checkpoint, self.localize_model_path)
            dist.barrier()


    def localize(self, rank, dataset):
        if not path.exists(self.localize_model_path):
            if rank == 0:
                print("No localization model")
            return

        device = torch.device(f"cuda:{rank}") if self.conf["train"]["cuda"] else torch.device("cpu")
        pinn_ckpt = torch.load(self.pinn_model_path)
        self.pinn_model.load_state_dict(pinn_ckpt["model"])
        localize_chpt = torch.load(self.localize_model_path)
        self.localize_model.load_state_dict(localize_chpt["model"])
        self.pinn_model = self.pinn_model.to(device)
        self.localize_model = self.localize_model.to(device)
        self.pinn_model.eval()
        self.localize_model.eval()

        loader = self._init_ddp(rank, [dataset], [False])[0]
        predicted = []
        if rank == 0:
            data_loader = tqdm(loader)
        else:
            data_loader = loader
        with torch.no_grad():
            for data in data_loader:
                data = data.to(device)
                embedding = self.pinn_model(data)
                data.x = embedding
                logits = self.localize_model(data)
                prediction = torch.sigmoid(logits)
                predicted.append(prediction)