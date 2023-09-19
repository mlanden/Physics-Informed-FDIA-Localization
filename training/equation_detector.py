from os import path
import numpy as np
from sklearn.metrics import confusion_matrix

from equations import build_equations

class EquationDetector():

    def __init__(self, conf, categorical_values, continuous_values) -> None:
        super().__init__()
        self.checkpoint_dir = path.join(conf["train"]["checkpoint_dir"], conf["train"]["checkpoint"])
        self.results_path = path.join("results", conf["train"]["checkpoint"])
        self.equations = build_equations(conf, categorical_values, continuous_values)

        self.residuals = [[] for _ in self.equations]
        self.deltas = [None for _ in self.equations]
        self.cumsum_states = [0 for _ in self.equations]
        self.thresholds = [0 for _ in self.equations]
        self.attacks = []
        self.alerts = []

    def training_step(self, batch):
        unscaled_seq, scaled_seq, target = batch

        for i in range(len(self.equations)):
            loss = self.equations[i].evaluate(unscaled_seq)
            self.residuals[i].append(loss)

    def detect(self, state):
        if any(delta is None for delta in self.deltas):
            for i in range(len(self.equations)):
                mean = np.mean(self.residuals[i])
                std = np.std(self.residuals[i])
                self.deltas[i] = mean + 2 * std
            print(self.deltas)
        unscaled_seq, scaled_seq, target, attack = state

        for i in range(len(self.equations)):
            residual = self.equations[i].evaluate(unscaled_seq)
            self.cumsum_states[i] = max(0, self.cumsum_states[i] + residual - self.deltas[i])
    
        # if attack:
        #     print(self.cumsum_states)
        alert = False
        for i in range(len(self.equations)):
            if self.cumsum_states[i] > self.thresholds[i]:
                alert = True
                self.cumsum_states[i] = 0
        self.attacks.append(attack)
        self.alerts.append(alert)

    def print_stats(self):
        cm = confusion_matrix(self.attacks, self.alerts)
        tp = cm[1][1]
        fp = cm[0][1]
        fn = cm[1][0]
        tn = cm[0][0]

        tpr = tp / (tp + fn)
        tnr = tn / (tn + fp)
        fpr = fp / (fp + tn)
        fnr = fn / (fn + tp)
        recall = tpr
        precision = tp / (tp + fp)
        f1 = 2 * precision * recall / (precision + recall)
        accuracy = (tp + tn) / len(self.attacks)
        print(f"True Positive: {tpr * 100 :3.2f}")
        print(f"True Negative: {tnr * 100 :3.2f}")
        print(f"False Positive: {fpr * 100 :3.2f}")
        print(f"False Negatives: {fnr * 100 :3.2f}")
        print(f"F1 Score: {f1 * 100 :3.2f}")
        print(f"Precision: {precision * 100 :3.2f}")
        print(f"Accuracy: {accuracy * 100 :3.2f}")

        dwells = []
        fns = []
        attack = False
        detect = False
        start = 0
        for i in range(len(self.attacks)):
            if self.attacks[i] and not self.alerts[i]:
                fns.append(i)
            if self.attacks[i] and not attack:
                attack = True
                start = i

            if attack and not detect and self.alerts[i]:
                dwells.append(i - start)
                attack = False
                detect = True
        dwell_time = np.mean(dwells)
        print(f"Dwell time: {dwell_time :.2f}")