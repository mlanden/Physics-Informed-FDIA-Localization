from typing import Tuple, Union
import torch
import torch.nn as nn
from torch.nn.parallel.distributed import DistributedDataParallel as DDP


def swat_loss(model, batch: torch.Tensor, target: torch.Tensor, categorical_values: dict, hidden_states=None) -> Union[
    Tuple[torch.Tensor, Tuple], torch.Tensor]:
    if hidden_states is not None:
        outputs, hidden_states = model.forward(batch, hidden_states)
    else:
        outputs = model.forward(batch)

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
