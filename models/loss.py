from typing import Tuple, Union, List
import torch
import torch.nn as nn
from functools import partial

from invariants import Invariant
from equations import Equation
from datasets import ICSDataset, SWATDataset


def prediction_loss(batch: torch.Tensor, outputs: List[torch.Tensor], target: torch.Tensor, categorical_values: dict) ->\
    torch.Tensor:
    losses = torch.zeros((batch.shape[0], batch.shape[-1]))
    continuous_idx = 0
    classification_idx = 0
    class_loss = nn.CrossEntropyLoss(reduction="none")
    continuous_loss = nn.MSELoss(reduction="none")
    for i in range(batch.shape[-1]):
        if i in categorical_values:
            continue
            # Cross entropy_loss
            logits = outputs[classification_idx + 1]
            target_class = target[:, i].long()
            # swat specific
            if categorical_values[i] == 2:
                target_class[:] -= 1

            losses[:, i] = class_loss(logits, target_class)
            classification_idx += 1
        else:
            # MSE loss
            predicted = outputs[0][:, continuous_idx]
            target_value = target[:, continuous_idx] - batch[:, -1, continuous_idx]
            losses[:, i] = continuous_loss(predicted, target_value)
            continuous_idx += 1
    # print(losses[0])
    return losses


def invariant_loss(batch: torch.Tensor, outputs: List[torch.Tensor], target: torch.Tensor, categorical_values: dict,
                   invariants: List[Invariant]=None) -> torch.Tensor:

    loss = torch.zeros((len(invariants), batch.shape[0]))
    for i, invariant in enumerate(invariants):
        confidence = invariant.confidence_loss(batch, outputs)
        # print(confidence, invariant.support)

        loss[i, :] = confidence #* invariant.support

    loss = torch.t(loss)
    return loss


def equation_loss(batch: torch.Tensor, outputs: List[torch.Tensor], target: torch.Tensor, categorical_values: dict,
                  equations: List[Equation]) -> torch.Tensor:
    loss = torch.zeros((len(equations), batch.shape[0]))
    for i, equation in enumerate(equations):
        confidence_loss = equation.confidence_loss(batch, outputs, target)
        loss[i, :] = confidence_loss
    loss = torch.t(loss)
    return loss
