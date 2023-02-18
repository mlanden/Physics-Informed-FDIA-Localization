from typing import List
import numpy as np
import torch
import torch.nn.functional as F
from .predicate import Predicate


class CategoricalPredicate(Predicate):

    def __init__(self, idx: int, categorical_idx: int, class_value: int):
        super().__init__()
        self.categorical_idx = categorical_idx
        self.idx = idx
        self.class_value = class_value

        self.loss = torch.nn.CrossEntropyLoss(reduction="none")

    def is_satisfied(self, states, network_outputs=None) -> bool:
        if network_outputs is None:
            return states[:, self.idx] == self.class_value
        else:
            categorical_output = network_outputs[self.categorical_idx + 1]
            class_id = torch.max(categorical_output, dim=1).indices
            return class_id == self.class_value

    def confidence(self, input_states, network_outputs: List[torch.Tensor]) -> torch.Tensor:
        categorical_output = network_outputs[self.categorical_idx + 1]

        target = torch.full((categorical_output.shape[0], ), self.class_value, device=categorical_output.device)
        loss = F.cross_entropy(categorical_output, target, reduction="none")
        return loss.view(1, -1)

    def __hash__(self):
        if self.hash is None:
            self.hash = hash((self.idx, self.class_value))
        return self.hash

    def __eq__(self, other):
        if not isinstance(other, CategoricalPredicate):
            return False

        return self.idx == other.idx and self.class_value == other.class_value

    def __str__(self):
        out = f"Categorical: Index:{self.idx}, Value: {self.class_value}"
        return out
