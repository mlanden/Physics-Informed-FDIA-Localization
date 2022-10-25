import queue
from typing import Tuple, Union, List
import torch
import torch.nn as nn
from torch import distributed as dist
from functools import partial
import multiprocessing as mp

from invariants import Invariant
from datasets import ICSDataset, SWATDataset


def swat_loss(batch: torch.Tensor, outputs: torch.Tensor, target: torch.Tensor, categorical_values: dict) -> Union[
    Tuple[torch.Tensor, Tuple], torch.Tensor]:
    losses = torch.zeros(target.shape[-1])
    continuous_idx = 0
    classification_idx = 0
    class_loss = nn.CrossEntropyLoss()
    continuous_loss = nn.MSELoss()
    for i in range(target.shape[-1]):
        if i in categorical_values:
            # Cross entropy_loss
            logits = outputs[classification_idx + 1]
            target_class = target[:, i].long()
            if categorical_values[i] == 2:
                target_class[:] -= 1

            losses[i] = class_loss(logits, target_class)
            classification_idx += 1
        else:
            # MSE loss
            predicted = outputs[0][:, continuous_idx]
            target_value = target[:, continuous_idx] - batch[:, -1, continuous_idx]
            losses[i] = continuous_loss(predicted, target_value)
            continuous_idx += 1

    return losses


def invariant_loss(batch: torch.Tensor, outputs: torch.Tensor, target: torch.Tensor, categorical_values: dict,
                   invariants: List[Invariant] = None, n_workers = 1) -> Union[Tuple[torch.Tensor, Tuple], torch.Tensor]:
    losses = torch.zeros((len(invariants)))
    for i, invariant in enumerate(invariants):
        losses[i] = invariant.confidence(batch, outputs)
        # if not dist.is_initialized() or dist.get_rank() == 0:
            # print("\r", end="")
            # print(f"{i + 1} / {len(invariants)} invariants evaluated", flush=True, end="")
    if not dist.is_initialized() or dist.get_rank() == 0:
        print()

    return losses


def get_losses(dataset: ICSDataset, invariants, n_workers = 1):
    loss_fns = []
    if isinstance(dataset, SWATDataset):
        loss_fns.append(swat_loss)
    else:
        raise RuntimeError("Unknown model type")

    if invariants is not None:
        loss_fns.append(partial(invariant_loss, invariants=invariants, n_workers=n_workers))

    return loss_fns
