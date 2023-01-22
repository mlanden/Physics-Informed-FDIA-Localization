import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("agg")
import json

if __name__ == '__main__':
    with open("fp_invariants.json", "r") as fd:
        fps = json.load(fd)

    fig, ax = plt.subplots()
    ax.boxplot(fps)
    plt.savefig("Invariant_false_positive.png")