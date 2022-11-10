import queue
from typing import List
import json
from os import path
import numpy as np


def save_results(tp, tn, fp, fn, labels, results_path, scores=None, delays=None):
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)

    print("\n")
    print(f"True Positive: {tpr * 100 :3.2f}")
    print(f"True Negative: {tnr * 100 :3.2f}")
    print(f"False Positive: {fpr * 100 :3.2f}")
    print(f"False Negatives: {fnr * 100 :3.2f}")
    results = {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "labels": labels
    }
    if scores is not None:
        results["scores"] = scores

    if delays is not None:
        results["delay"] = delays
        print(f"Delay: {np.mean(delays) :.2f}")

    with open(path.join(results_path, "detection.json"), "w") as fd:
        json.dump(results, fd)


