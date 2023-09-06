import os
from os import path
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


if __name__ == '__main__':
    # files = ["noinvariant_mean", "noinvariant_gmm", "invariant_mean", "updated", "fp_reduce"]
    # names = ["Z Score, No invariant", "GMM No Invariant", "Z Score Invariant", "Updated Labels", "Reduced Invariants"]
    experiments = ["swat_benchmark", "swat_structure", "swat_optimized", "swat_lstm", "swat_dropout", "swat_autoencoder_equ"]
    names = ["Swat", "Swat structure", "Swat Stacked", "SWaT LSTM", "SWaT Dropout", "SWaT autoencoder"]
    fig = plt.figure()
    for experiment, name in zip(experiments, names):
        with open(f"../{experiment}/evaluation_losses.json", "r") as fd:
            data = json.load(fd)

        labels = []
        scores = []
        for score, label in data:
            scores.append(score)
            labels.append(label)

        fpr, tpr, thresholds = roc_curve(labels, scores)
        plt.plot(fpr, tpr, label=name)
        print()
        print()
        print(name)
        for i in range(len(tpr)):
            if fpr[i] < .1 and tpr[i] > .7:
                print(f"{i}, {tpr[i]:0.2f}, {fpr[i]:0.2f}, {thresholds[i]:.2f}")
        # print(tpr.tolist())
        # print(thresholds.tolist())
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.xticks(np.arange(0, 1.1, .1))
    plt.yticks(np.arange(0, 1.1, .1))
    plt.legend()
    plt.grid(True)
    plt.savefig("roc.png")
    # plt.show()