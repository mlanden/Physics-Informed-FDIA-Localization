from typing import Tuple, Union, List
import torch
import torch.nn as nn
from functools import partial

from invariants import Invariant
from datasets import ICSDataset, SWATDataset


def swat_loss(model, batch: torch.Tensor, target: torch.Tensor, categorical_values: dict, hidden_states=None) -> Union[
    Tuple[torch.Tensor, Tuple], torch.Tensor]:
    hidden_states, outputs = evaluate_model(batch, hidden_states, model)

    losses = torch.zeros(batch.shape[-1])
    continuous_idx = 0
    classification_idx = 0
    class_loss = nn.CrossEntropyLoss()
    continuous_loss = nn.MSELoss()
    for i in range(batch.shape[-1]):
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

    if hidden_states is not None:
        return losses, hidden_states
    else:
        return losses


def evaluate_model(batch, hidden_states, model):
    if hidden_states is not None:
        outputs, hidden_states = model.forward(batch, hidden_states)
    else:
        outputs = model.forward(batch)
    return hidden_states, outputs


def invariant_loss(model, batch: torch.Tensor, target: torch.Tensor, categorical_values: dict, hidden_states=None, invariants: List[Invariant]=None) -> Union[
    Tuple[torch.Tensor, Tuple], torch.Tensor]:
    hidden_states, outputs = evaluate_model(batch, hidden_states, model)

    loss = torch.zeros((len(invariants)))
    for i, invariant in enumerate(invariants):
        loss[i] = invariant.confidence(outputs)
        print("\r", end="")
        print(f"{i} / {len(invariants)} invariants evaluated", end="")
    print()

    if hidden_states is not None:
        return loss, hidden_states
    else:
        return loss


def get_losses(dataset: ICSDataset, invariants):
    loss_fns = []
    if isinstance(dataset, SWATDataset):
        loss_fns.append(swat_loss)
    else:
        raise RuntimeError("Unknown model type")

    if invariants is not None:
        loss_fns.append(partial(invariant_loss, invariants=invariants))

    return loss_fns