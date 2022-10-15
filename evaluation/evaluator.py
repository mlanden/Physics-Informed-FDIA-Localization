from abc import ABC, abstractmethod
from os import path
import json
import numpy as np

from torch.utils.data import DataLoader


class Evaluator(ABC):
    def __init__(self, conf, dataset):
        self.dataset = DataLoader(dataset)
        self.conf = conf
        self.checkpoint = conf["train"]["checkpoint"]
        self.results_path = path.join("results", self.checkpoint)

    @abstractmethod
    def alert(self, state, target):
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
        for features, target, attack in self.dataset:
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

            alert = self.alert(features, target)
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
            msg = f"{step :5d} / {len(self.dataset)}: TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}"
            if len(delays) > 0:
                msg += f", Dwell: {np.mean(delays):.3f}"
            print(f"\r", msg, end="")
        print()

        tpr = tp / (tp + fn)
        tnr = tn / (tn + fp)
        fpr = fp / (fp + tn)
        fnr = fn / (fn + tp)

        print(f"True Positive: {tpr * 100 :3.2f}")
        print(f"True Negative: {tnr * 100 :3.2f}")
        print(f"False Positive: {fpr * 100 :3.2f}")
        print(f"False Negatives: {fnr * 100 :3.2f}")

        results = {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "delay": delays,
            "scores": scores,
            "labels": labels
        }
        with open(path.join(self.results_path, "detection.json"), "w") as fd:
            json.dump(results, fd)
