import json
import  os
import tempfile
from os import path
import ray
from ray import train
from sklearn.metrics import multilabel_confusion_matrix, precision_score, recall_score, f1_score, roc_curve
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
from utils import EarlyStopping, early_stopping


class LocalizationTrainer:

    def __init__(self, conf):
        self.conf = conf
        self.batch_size = conf["train"]["batch_size"]
        self.checkpoint_dir = path.join(conf["train"]["checkpoint_dir"], conf["train"]["checkpoint"])
        self.pinn_model_path = path.join(self.checkpoint_dir, "pinn.pt")
        self.model_name = "localize.pt"
        self.localize_model_path = path.join(self.checkpoint_dir, self.model_name)
        self.size = self.conf["train"]["gpus"]
        self.scale = conf["train"]["scale"]
        self.equations = build_equations(conf)
        self.localize_model = GCN(self.conf, 2, 2)

    def _init_ddp(self, rank, datasets, shuffles):
        if not ray.is_initialized():
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = '29508'
            if self.conf["train"]["cuda"]:
                backend = "nccl"
            else:
                backend = "gloo"
            dist.init_process_group(backend, rank=rank, world_size=self.size)
            if backend == "gloo":
                self.localize_model = torch.nn.parallel.DistributedDataParallel(self.localize_model)
            else:
                self.localize_model = torch.nn.parallel.DistributedDataParallel(self.localize_model,
                                                                                device_ids=[rank],
                                                                                output_device=rank)
        else:
            if not train.get_checkpoint():
                self.localize_model = train.torch.prepare_model(self.localize_model)

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

        optim = torch.optim.Adam(self.localize_model.parameters(), 
                                 lr=self.conf["train"]["lr"],
                                 weight_decay=self.conf["train"]["regularization"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.5)
        
        device = torch.device(f"cuda:{rank}") if self.conf["train"]["cuda"] else torch.device("cpu")
        self.localize_model = self.localize_model.to(device)
        checkpoint = None
        if ray.ray.is_initialized():
            checkpoint = train.get_checkpoint()
            if checkpoint:
                with checkpoint.as_directory() as checkpoint_dir:
                    checkpoint = torch.load(path.join(checkpoint_dir, self.model_name))
        elif self.conf["train"]["load_checkpoint"] and path.exists(self.localize_model_path):
            checkpoint = torch.load(self.localize_model_path)
        
        if checkpoint is not None:
            start_epoch = checkpoint["epoch"]
            self.localize_model.load_state_dict(checkpoint["model"])
            optim.load_state_dict(checkpoint["optim"])
            scheduler.load_state_dict(checkpoint["schedule"])
            best_loss = checkpoint["loss"]

        train_loader, val_loader = self._init_ddp(rank, [train_dataset, val_dataset], [True, False])
        rank = dist.get_rank()
        if rank == 0:
            print(self.localize_model)
        epochs = self.conf["train"]["epochs"]

        if rank == 0 and not ray.is_initialized():
            tb_dir = path.join(self.checkpoint_dir, "localization_training")
            if not path.exists(tb_dir):
                os.makedirs(tb_dir)
            version = len([name for name in os.listdir(tb_dir) if path.isdir(path.join(tb_dir, name))]) + 1
            print("Version:", version)
            tb_writer = SummaryWriter(path.join(tb_dir, f"version_{version}"))

        early_stopping = EarlyStopping(20, 0.0001)
        torch.autograd.set_detect_anomaly(True)
        for epoch in range(start_epoch, epochs):
            self.localize_model.train()
            train_loader.sampler.set_epoch(epoch)
            if rank == 0 and not ray.is_initialized():
                loader = tqdm(train_loader, position=0, leave=True)
            else:
                loader = train_loader
            total_loss = 0
            total_physics_loss = 0
            total_localization_loss = 0
            for data in loader:
                optim.zero_grad()
                data = data.to(device)
                loss, localization_loss, physics_loss = self.compute_loss(data)
            
                loss.backward()
                if self.conf["train"]["max_norm"] > 0:
                    torch.nn.utils.clip_grad.clip_grad_norm_(self.localize_model.parameters(), self.conf["train"]["max_norm"])
                optim.step()
                total_loss += loss.detach()
                total_physics_loss += physics_loss.detach()
                total_localization_loss += localization_loss.detach()

                if rank == 0 and not ray.is_initialized():
                    loader.set_description(f"Train Epoch [{epoch: 3d}/{epochs}]")
                    loader.set_postfix(loss=loss.item(), physics_loss=physics_loss.item(), 
                                       localization_loss=localization_loss.item())
            total_loss /= len(train_loader)
            total_physics_loss /= len(train_loader)
            total_localization_loss /= len(train_loader)
            dist.all_reduce(total_loss)
            dist.all_reduce(total_physics_loss)
            dist.all_reduce(total_localization_loss)

            total_loss /= self.size
            total_localization_loss /= self.size
            total_physics_loss /= self.size
            if rank == 0 and not ray.is_initialized():
                tb_writer.add_scalar("Training loss", total_loss, epoch)
                tb_writer.add_scalar("Physics loss", total_physics_loss, epoch)
                tb_writer.add_scalar("Localization loss", total_localization_loss, epoch)

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
                    loss, localization_loss, physics_loss = self.compute_loss(data)
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

                    if loss < best_loss or ray.is_initialized():
                        try:
                            model = self.localize_model.module
                        except AttributeError:
                            model = self.localize_model
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
  
                if early_stopping.stop_training(total_loss.item()):
                    break
            dist.barrier()


    def localize(self, rank, dataset):
        if not path.exists(self.localize_model_path):
            if rank == 0:
                print("No localization model")
            return

        device = torch.device(f"cuda:{rank}") if self.conf["train"]["cuda"] else torch.device("cpu")
        
        localize_chpt = torch.load(self.localize_model_path)
        self.localize_model.load_state_dict(localize_chpt["model"])
        self.localize_model = self.localize_model.to(device)
        self.localize_model.eval()

        loader = self._init_ddp(rank, [dataset], [False])[0]
        predicted = []
        truth = []
        thresholds = []
        ids = []
        if rank == 0:
            data_loader = tqdm(loader)
        else:
            data_loader = loader
        with torch.no_grad():
            for data in data_loader:
                data = data.to(device)
                _, logits = self.localize_model(data)
                threshold = torch.sigmoid(logits + 1e-10)
                prediction = threshold > 0.5
                thresholds.append(threshold)
                predicted.append(prediction.float())
                truth.append(data.classes)
                ids.append(data.idx.float())
            thresholds = torch.cat(thresholds)
            predicted = torch.cat(predicted)
            truth = torch.cat(truth)
            ids = torch.cat(ids)

            if rank == 0:
                if self.size > 1:
                    all_predicted = [torch.empty(predicted.size(), device=device)
                                    for _ in range(self.size)]
                    dist.gather(predicted, all_predicted, 0)
                    predicted = torch.cat(all_predicted).cpu()
                    all_truth = [torch.empty(truth.size(), device=device)
                                for _ in range(self.size)]
                    dist.gather(truth, all_truth, 0)
                    truth = torch.cat(all_truth).cpu()
                    all_thresholds = [torch.empty(thresholds.size(), device=device)
                                  for _ in range(self.size)]
                    dist.gather(thresholds, all_thresholds, 0)
                    thresholds = torch.cat(all_thresholds).cpu()
                    all_ids = [torch.empty(ids.size(), device=device)
                               for _ in range(self.size)]
                    dist.gather(ids, all_ids, 0)
                    ids = torch.cat(all_ids).cpu()
                else:
                    truth = truth.cpu()
                    predicted = predicted.cpu()
                    thresholds = thresholds.cpu()
                    ids = ids.cpu()

                true_y = truth.flatten()
                scores = thresholds.flatten()
                fpr, tpr, thresholds = roc_curve(true_y, scores)
                plt.plot(fpr, tpr)
                plt.grid(True)
                plt.savefig("roc.png")

                missed = torch.nonzero((truth == 1) & (predicted == 0))
                missed_grids = ids[missed[:, 0]].numpy().tolist()
                # with open("missed_grids_pinn.json", "w") as fd:
                #     json.dump(missed_grids, fd)

                recall = recall_score(truth, predicted, average=self.conf["train"]["average"])
                precision = precision_score(truth, predicted, average=self.conf["train"]["average"])
                f1score = f1_score(truth, predicted, average=self.conf["train"]["average"])

                confusion_matrixs = multilabel_confusion_matrix(truth, predicted)
                false_positive_rates = []
                for i in range(len(confusion_matrixs[0])):
                    tn, fp, fn, tp = confusion_matrixs[i].ravel()
                    fpr = fp / (fp + tn) if fp + tn != 0 else 0
                    false_positive_rates.append(fpr)
                avg_fpr = np.mean(false_positive_rates)
                print(f"Recall: {recall * 100}")
                print(f"Precision: {precision * 100}")
                print(f"F1 score: {f1score * 100}")
                print(f"False positive rate: {avg_fpr * 100}")
            elif self.size > 1:
                dist.gather(predicted, [], 0)
                dist.gather(truth, [], 0)
                dist.gather(thresholds, [], 0)
                dist.gather(ids, [], 0)

    def compute_loss(self, data):
        pinn_output, localization_output = self.localize_model(data)
        localization_loss = F.binary_cross_entropy_with_logits(localization_output, data.classes)

        physics_loss = torch.zeros((len(data.x), len(self.equations)), 
                                   device=pinn_output.device)
        for i, equation in enumerate(self.equations):
            physics_loss[:, i] = equation.confidence_loss(data, pinn_output, None)
        physics_loss = physics_loss.mean()
        localization_loss = localization_loss.mean()
        loss = self.scale[0] * localization_loss + self.scale[1] * physics_loss
        return loss, localization_loss, physics_loss