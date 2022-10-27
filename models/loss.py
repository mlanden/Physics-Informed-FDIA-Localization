from typing import Tuple, Union, List
import torch
import torch.nn as nn
from functools import partial

from invariants import Invariant
from datasets import ICSDataset, SWATDataset


def prediction_loss(model, batch: torch.Tensor, target: torch.Tensor, categorical_values: dict, hidden_states=None) -> Union[
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

    return losses, hidden_states


def evaluate_model(batch, hidden_states, model):
    outputs, hidden_states = model.forward(batch, hidden_states)
    return hidden_states, outputs


def invariant_loss(model, batch: torch.Tensor, target: torch.Tensor, categorical_values: dict, hidden_states=None, invariants: List[Invariant]=None) -> Union[
    Tuple[torch.Tensor, Tuple], torch.Tensor]:
    hidden_states, outputs = evaluate_model(batch, hidden_states, model)

    loss = torch.zeros((len(invariants)))
    for i, invariant in enumerate(invariants):
        loss[i] = invariant.confidence(batch, outputs)
    #     print("\r", end="")
    #     print(f"{i} / {len(invariants)} invariants evaluated", end="")
    # print()

    return loss, hidden_states


def get_losses(invariants):
    loss_fns = [prediction_loss]

    if invariants is not None:
        loss_fns.append(partial(invariant_loss, invariants=invariants))

    return loss_fns