import json
from os import path
import numpy as np
from sklearn.metrics import roc_curve
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt


def make_roc_curve(eval_file):
    with open(eval_file, "r") as fd:
        data = json.load(fd)

    labels = []
    scores = []
    for score, label in data:
        scores.append(score)
        labels.append(label)

    fpr, tpr, thresholds = roc_curve(labels, scores)

    fig = plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig("mean invarants.png")
