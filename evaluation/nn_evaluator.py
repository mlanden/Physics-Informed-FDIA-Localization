from os import path
import torch

from datasets import SWATDataset
from models import swat_loss
from .evaluator import Evaluator
from datasets import ICSDataset


class NNEvaluator(Evaluator):
    def __init__(self, conf: dict, dataset: ICSDataset):
        super(NNEvaluator, self).__init__(conf, dataset)
        print(f"Number of samples: {len(dataset)}")

        self.model_path = path.join("checkpoint", self.checkpoint, "model.pt")
        self.normal_behavior_path = path.join("checkpoint", self.checkpoint, "normal_behavior.pt")

        info = torch.load(self.model_path)
        self.model = info["model"]
        self.model.eval()
        obj = torch.load(self.normal_behavior_path)
        self.normal_means = obj["mean"]
        self.normal_stds = obj["std"]

        if isinstance(dataset, SWATDataset):
            self.loss_fn = swat_loss
        else:
            raise RuntimeError("Unknown model type")

        self.hidden_states = [None for _ in range(len(self.conf["model"]["hidden_layers"]) - 1)]

    def alert(self, state, target):
        state = state.unsqueeze(0)
        losses, self.hidden_states = self.loss_fn(self.model, state, target, self.categorical_features,
                                                  self.hidden_states)
        losses = losses.detach()
        # losses = torch.sum(losses)
        score = torch.abs(losses - self.normal_means) / self.normal_stds

        # scores.append(score.item())
        # print(loss, normal_means, normal_stds, attack)
        alarm = torch.any(score > 2)
        return alarm
