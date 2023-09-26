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
        self.hits = [0 for _ in self.equations]
        self.attacks = []
        self.alerts = []

    def training_step(self, batch):
        unscaled_seq, scaled_seq, target = batch

        for i in range(len(self.equations)):
            loss = self.equations[i].evaluate(unscaled_seq)
            self.residuals[i].append(loss)

    def detect(self, state, attack_idx):
        if any(delta is None for delta in self.deltas):
            means = [0 for _ in self.equations]
            stds = [0 for _ in self.equations]
            for i in range(len(self.equations)):
                means[i] = np.mean(self.residuals[i])
                stds[i] = np.std(self.residuals[i])
                self.deltas[i] = means[i] + 2 * stds[i]
            print("Mean:", means)
            print("Standard dev:", stds)
            print(self.deltas)
        unscaled_seq, scaled_seq, target, attack, attack_idx = state

        for i in range(len(self.equations)):
            residual = self.equations[i].evaluate(unscaled_seq)
            self.cumsum_states[i] = max(0, self.cumsum_states[i] + residual - self.deltas[i])
    
        alert = False
        for i in range(len(self.equations)):
            if self.cumsum_states[i] > self.thresholds[i]:
                alert = True
                self.cumsum_states[i] = 0
                self.hits[i] += 1
        self.attacks.append(attack)
        self.alerts.append(alert)

    def print_stats(self, attack_map):
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
        detected_attacks = []
        attack = False
        detect = False
        start = 0
        for i in range(len(self.attacks)):
            if not self.attacks[i]:
                attack = False

            if self.attacks[i] and not attack:
                attack = True
                start = i
                detect = False

            if attack and not detect and self.alerts[i]:
                print(i)
                dwells.append(i - start)
                detect = True
                for attack in attack_map:
                    if i in attack_map[attack]:
                        detected_attacks.append(attack)
        
        dwell_time = np.mean(dwells)
        print(f"Dwell time: {dwell_time :.2f}")
        print("Detected attacks:", detected_attacks)
        print("Hits:", self.hits)