import os
from os import path
import json
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


if __name__ == '__main__':
    print(os.getcwd())
    # files = ["noinvariant_mean", "noinvariant_gmm", "invariant_mean", "updated", "fp_reduce"]
    # names = ["Z Score, No invariant", "GMM No Invariant", "Z Score Invariant", "Updated Labels", "Reduced Invariants"]
    files = ["prediction", "fp_reduce", "gmm"]
    names = ["LSTM Prediction", "Reduced Invariants", "GMM"]
    fig = plt.figure()
    for i in range(len(files)):
        with open(f"evaluation_losses_{files[i]}.json", "r") as fd:
            data = json.load(fd)

        labels = []
        scores = []
        for score, label in data:
            scores.append(score)
            labels.append(label)

        fpr, tpr, thresholds = roc_curve(labels, scores)
        plt.plot(fpr, tpr, label=names[i])
        print()
        print()
        print(names[i])
        for i in range(len(tpr)):
            if .70 < tpr[i] < .75:
                print(f"{i}, {tpr[i]:0.2f}, {fpr[i]:0.2f}, {thresholds[i]:.2f}")
        print(tpr.tolist())
        # print(thresholds.tolist())
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    # plt.savefig("roc.png")
    plt.show()