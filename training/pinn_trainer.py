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

from models import FCN, GCN
from equations import build_equations

class PINNTrainer:

    def __init__(self, conf, dataset) -> None:
        self.conf = conf
        self.batch_size = conf["train"]["batch_size"]
        self.scale = conf["train"]["scale"]
        self.use_physics = conf["train"]["physics"]
        self.use_graph = conf["model"]["graph"]
        self.model_name = "pinn.pt"

        self.n_buses = conf["data"]["n_buses"]
        self.checkpoint_dir = path.join(conf["train"]["checkpoint_dir"], conf["train"]["checkpoint"])
        self.checkpoint_path = path.join(self.checkpoint_dir, self.model_name)
        self.size = self.conf["train"]["gpus"]

        if self.use_graph:
            self.model = GCN(conf, 2, 2)
        else:
            self.model = FCN(conf, 2 * self.n_buses + 2 * self.n_buses ** 2, 2 * self.n_buses)
        self.equations = build_equations(conf, dataset.get_categorical_features(), 
                                         dataset.get_continuous_features())
        
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
                self.model = torch.nn.parallel.DistributedDataParallel(self.model)
            else:
                self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[rank], output_device=rank)
        else:
            if not train.get_checkpoint():
                self.model = train.torch.prepare_model(self.model)

        loaders = []
        for dataset, shuffle in zip(datasets, shuffles):
            sampler = DistributedSampler(dataset,
                                         num_replicas=self.size,
                                         rank=rank,
                                         shuffle=shuffle)
            if self.use_graph:
                loader = gDataLoader(dataset,
                                   batch_size=self.batch_size,
                                   shuffle=False,
                                   sampler=sampler)
            else:
                loader = DataLoader(dataset,
                                    batch_size=self.batch_size,
                                    shuffle=False,
                                    sampler=sampler)
            loaders.append(loader)
        return loaders

    def train(self, rank, train_dataset, val_dataset):
        start_epoch = 0
        best_loss = torch.inf
        optim = torch.optim.Adam(self.model.parameters(), lr=self.conf["train"]["lr"],
                                  weight_decay=self.conf["train"]["regularization"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.5)
        device = torch.device(f"cuda:{rank}") if self.conf["train"]["cuda"] else torch.device("cpu")
        self.model = self.model.to(device)

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
            tb_dir = path.join(self.checkpoint_dir, "pinn_training")
            if not path.exists(tb_dir):
                os.makedirs(tb_dir)
            version = len([name for name in os.listdir(tb_dir) if path.isdir(path.join(tb_dir, name))]) + 1
            print("Version:", version)
            tb_writer = SummaryWriter(path.join(tb_dir, f"version_{version}"))
        
        for epoch in range(start_epoch, epochs):
            self.model.train()
            train_loader.sampler.set_epoch(epoch)
            if rank == 0 and not ray.is_initialized():
                loader = tqdm(train_loader, position=0, leave=True)
            else:
                loader = train_loader
            total_loss = 0

            for data in loader:
                optim.zero_grad()
                if not self.use_graph:
                    inputs, targets = data
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                else:
                    data = data.to(device)
                    inputs = data
                    targets = data.y
                data_loss, physics_loss, loss = self._combine_losses(inputs, targets)
                loss.backward()
                if self.conf["train"]["max_norm"] > 0:
                    torch.nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), self.conf["train"]["max_norm"])
                optim.step()
                total_loss += loss.detach()
                if rank == 0 and not ray.is_initialized():
                    loader.set_description(f"Train Epoch [{epoch: 3d}/{epochs}]")
                    if self.use_physics:
                        loader.set_postfix(loss=loss.item(), data_loss=data_loss.item(), physics_loss=physics_loss.item())
                    else:
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

            self.model.eval()
            with torch.no_grad():
                total_data_loss = 0
                total_physics_loss = 0
                total_loss = 0
                for data in loader:
                    if not self.use_graph:
                        inputs, targets = data
                        inputs = inputs.to(device)
                        targets = targets.to(device)
                    else:
                        data = data.to(device)
                        inputs = data
                        targets = data.y
                    data_loss, physics_loss, loss = self._combine_losses(inputs, targets)
                    total_data_loss += data_loss
                    total_physics_loss += physics_loss
                    total_loss += loss

                    if rank == 0 and not ray.is_initialized():
                        loader.set_description(f"Val Epoch [{epoch: 3d}/{epochs}]")
                        if self.use_physics:
                           loader.set_postfix(loss=loss.item(), data_loss=data_loss.item(), physics_loss=physics_loss.item())
                        else:
                            loader.set_postfix(loss=loss.item())

                total_data_loss /= len(val_loader)
                total_physics_loss /= len(val_loader)
                total_loss /= len(val_loader)

                dist.all_reduce(total_data_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_physics_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
                total_data_loss /= self.size
                total_physics_loss /= self.size
                total_loss /= self.size

                scheduler.step(total_loss)

                if rank == 0:
                    lr = optim.param_groups[0]['lr']
                    if not ray.is_initialized():
                        tb_writer.add_scalar("Learning Rate", lr, epoch)
                        tb_writer.add_scalar("Validation Loss", total_loss, epoch)
                        if self.use_physics:
                            tb_writer.add_scalar("Data loss", total_data_loss, epoch)
                            tb_writer.add_scalar("Physics loss", total_physics_loss, epoch)

                    if loss < best_loss or ray.is_initialized() or 1:
                        try:
                            model = self.model.module
                        except AttributeError:
                            model = self.model
                        checkpoint = {
                            "model": model.state_dict(),
                            "structure": self.conf["model"],
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
                            torch.save(checkpoint, self.checkpoint_path)
            dist.barrier()

    def create_normal_profile(self, rank, dataset):
    
        if not path.exists(self.checkpoint_path):
            if rank == 0:
                print("Checpoint does not exist")
            return
        device = torch.device(f"cuda:{rank}") if self.conf["train"]["cuda"] else torch.device("cpu")
        checkpoint = torch.load(self.checkpoint_path)
        if rank == 0:
            print(checkpoint["structure"])
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        if rank == 0:
            print(self.model)
        self.model = self.model.to(device)
        loader = self._init_ddp(rank, [dataset], [False])[0]

        losses = []
        if rank == 0:
            data_loader = tqdm(loader)
        else:
            data_loader = loader

        with torch.no_grad():
            for data in data_loader:
                if self.use_graph:
                    data = data.to(device)
                    inputs = data
                    targets = data.y
                else:
                    inputs, targets = data
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                data_loss, physics_loss = self._compute_loss(inputs, targets)
                data_loss = data_loss.mean(dim=1).view(-1, 1)
                # loss = torch.cat([data_loss, physics_loss], dim=1)
                loss = physics_loss
                losses.append(loss)
            loss = torch.cat(losses)

        if rank == 0:
            all_losses = [torch.empty(loss.size(), device=device) for _ in range(self.size)]
            dist.gather(loss, all_losses, 0)
            losses = torch.cat(all_losses).cpu()
            self.plot(losses)
            
            mean = torch.mean(losses, dim=0)
            std = torch.std(losses, dim=0)
            checkpoint["mean"] = mean
            checkpoint["std"] = std
            print("Mean:", mean)
            print("Std:", std)
            torch.save(checkpoint, self.checkpoint_path)
        else:
            dist.gather(loss, [], 0)
        dist.barrier()

    def detect(self, rank, datset):
        if not path.exists(self.checkpoint_path):
            if rank == 0:
                print("Checkpoint does not exist")
            return
        
        device = torch.device(f"cuda:{rank}") if self.conf["train"]["cuda"] else torch.device("cpu")
        checkpoint = torch.load(self.checkpoint_path)
        mean = checkpoint["mean"].to(device)
        std = checkpoint["std"].to(device)
        threshold = self.conf["model"]["threshold"]
        self.model.load_state_dict(checkpoint["model"])
        self.model = self.model.to(device)
        self.model.eval()
        loader = self._init_ddp(rank, [datset], [False])[0]
        if rank == 0:
            data_loader = tqdm(loader)
        else:
            data_loader = loader

        alarms = []
        attacks = []
        # attack_idxs = []
        scores = []
        max_idx = []
        with torch.no_grad():
            for data in data_loader:
                if not self.use_graph:
                    inputs, targets = data
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                else:
                    data = data.to(device)
                    inputs = data
                    targets = data.y
                    attack = data.label
                data_loss, physics_loss = self._compute_loss(inputs, targets)
                data_loss = data_loss.mean(dim=1).view(-1, 1)
                # loss = torch.cat([data_loss, physics_loss], dim=1)
                loss = physics_loss
                score = (loss - mean) / std
                maxes = torch.max(score, dim=1)
                scores.append(maxes.values)
                max_idx.append(maxes.indices)
                alarm = torch.max(score, dim=1).values > threshold
                
                alarms.append(alarm)
                attacks.append(attack)
        # print(max_idx)
        alarms = torch.cat(alarms)
        attacks = torch.cat(attacks)
        scores = torch.cat(scores).detach()
        # attack_idxs = torch.cat(attack_idxs)

        if rank == 0:
            all_alarms = [torch.empty_like(alarms) for _ in range(self.size)]
            all_attacks = [torch.empty_like(attacks) for _ in range(self.size)]
            # all_attack_idxs = [torch.empty_like(attack_idxs) for _ in range(self.size)]
            all_scores = [torch.empty_like(scores) for _ in range(self.size)]
            dist.gather(alarms, all_alarms, 0)
            dist.gather(attacks, all_attacks, 0)
            # dist.gather(attack_idxs, all_attack_idxs, 0)
            dist.gather(scores, all_scores, 0)
            alarms = torch.cat(all_alarms).cpu()
            attacks = torch.cat(all_attacks).cpu()
            # attack_idxs = torch.cat(all_attack_idxs)
            scores = torch.cat(all_scores).cpu()

            self._compute_stats(alarms, attacks, scores)#, attack_idxs)
        else:
            dist.gather(alarms, [], 0)
            dist.gather(attacks, [], 0)
            # dist.gather(attack_idxs, [], 0)
            dist.gather(scores, [], 0)
        dist.barrier()

    def _compute_stats(self, alarms, attacks, scores, attack_idxs=0):
        cm = confusion_matrix(attacks, alarms)
        tp = cm[1][1]
        fp = cm[0][1]
        fn = cm[1][0]
        tn = cm[0][0]

        tpr = tp / (tp + fn)
        tnr = tn / (tn + fp)
        fpr = fp / (fp + tn)
        fnr = fn / (fn + tp)
        recall = tpr
        precision = tp / (tp + fp)
        f1 = 2 * precision * recall / (precision + recall)
        accuracy = (tp + tn) / len(attacks)
        print(f"True Positive: {tpr * 100 :3.2f}")
        print(f"True Negative: {tnr * 100 :3.2f}")
        print(f"False Positive: {fpr * 100 :3.2f}")
        print(f"False Negatives: {fnr * 100 :3.2f}")
        print(f"F1 Score: {f1 * 100 :3.2f}")
        print(f"Precision: {precision * 100 :3.2f}")
        print(f"Accuracy: {accuracy * 100 :3.2f}")

        fpr, tpr, thresholds = roc_curve(attacks, scores)
        plt.plot(fpr, tpr)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.xticks(np.arange(0, 1.1, .1))
        plt.yticks(np.arange(0, 1.1, .1))
        plt.grid(True)
        plt.savefig(path.join(self.checkpoint_dir, "roc.png"))

        with open(path.join(self.checkpoint_dir, "thresholds.txt"), "w") as fd:
            for i in range(len(tpr)):
                if fpr[i] < .1 and tpr[i] > .7:
                    print(f"{i}, {tpr[i]:0.2f}, {fpr[i]:0.2f}, {thresholds[i]:.2f}", file=fd)

    def _combine_losses(self, inputs, targets):
        data_loss, physics_loss = self._compute_loss(inputs, targets)
        data_loss = data_loss.mean()
        physics_loss = physics_loss.mean()
        loss = self.scale[0] * data_loss + self.scale[1] * physics_loss
        return data_loss, physics_loss, loss

    def _compute_loss(self, inputs, targets):
        predicted = self.model(inputs)
        # data_loss = F.mse_loss(predicted, targets, reduction='none')
        data_loss = torch.zeros_like(targets)

        physics_loss = torch.zeros((len(inputs), len(self.equations)), 
                                   device=predicted.device)
        if self.use_physics:
            for i, equation in enumerate(self.equations):
                physics_loss[:, i] = equation.confidence_loss(inputs, predicted, targets)
        if self.use_graph:
            data_loss = data_loss.view(physics_loss.size(0), -1)
        return data_loss, physics_loss
    
    def plot(self, losses):
        losses = torch.flatten(losses)
        ax = plt.subplot()
        ax.hist(losses, bins=20, log=True)
        ax.set(title="Physics Losses",
               xlabel="Loss",
               ylabel="Count")
        plt.tight_layout()
        plt.savefig(path.join(self.checkpoint_dir, "loss_plot.png"))
