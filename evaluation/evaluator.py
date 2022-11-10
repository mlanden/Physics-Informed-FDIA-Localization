import time
from abc import ABC, abstractmethod
from tqdm import tqdm
from os import path
import json
import numpy as np

from torch.utils.data import DataLoader

from datasets import ICSDataset
from utils import save_results


class Evaluator(ABC):
    def __init__(self, conf: dict, dataset: ICSDataset):
        self.categorical_features = dataset.get_categorical_features()
        self.dataset = dataset
        self.conf = conf
        self.checkpoint_dir = path.join("checkpoint", conf["train"]["checkpoint"])
        self.results_path = path.join("results", conf["train"]["checkpoint"])

    @abstractmethod
    def alert(self, state, target, attack):
        pass

    def close(self):
        pass

    def evaluate(self):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        delays = []
        scores = []
        labels = []

        attack_start = -1
        attack_detected = False

        step = 0
        for features, target, attack in DataLoader(self.dataset):
            # if step == 5000:
            #     break
            if attack_start > -1 and not attack:
                attack_start = -1
                if not attack_detected:
                    # Did not detect attack
                    fn += 1
                attack_detected = False

            if attack_detected:
                # Already detected attack
                continue

            labels.append(1 if attack else 0)
            if attack_start == -1 and attack:
                attack_start = step

            alert = self.alert(features, target, attack)
            if attack:
                if alert:
                    delay = step - attack_start
                    delays.append(delay)
                    tp += 1
                    attack_detected = True
            else:
                if alert:
                    fp += 1
                else:
                    tn += 1
            step += 1

            msg = f"{step :5d} / {len(self.dataset)}:"
            if tp + fn > 0:
                tpr = tp / (tp + fn)
                msg += f" TPR: {tpr * 100: 3.2f}"
            else:
                msg += " TPR: --"
            if tn + fp > 0:
                tnr = tn / (tn + fp)
                msg += f", TNR: {tnr * 100: 3.2f}"
            else:
                msg += ", TNR: --"
            if fp + tn > 0:
                fpr = fp / (fp + tn)
                msg += f", FPR: {fpr * 100: 3.2f}"
            else:
                msg += ", FPR: --"
            if fn + tp > 0:
                fnr = fn / (fn + tp)
                msg += f", FNR: {fnr * 100:3.2f}"
            else:
                msg += ", FNR: --"
            if len(delays) > 0:
                msg += f", Dwell: {np.mean(delays):.3f}"
            print(f"\r", msg, end="")
            # if step == 10000:
            #     break

        print()
        self.on_evaluate_end()
        self.close()

        save_results(tp, tn, fp, fn, labels, self.results_path, scores, delays)

    def on_evaluate_end(self):
        pass
