import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

if __name__ == "__main__":
    pinn_file = "missed_grids_pinn.json"
    arma_file = "missed_grids_arma.json"
    training_data = "../data/grid/14_perfect_fdia.csv"
    training_data = pd.read_csv(training_data)
    features = training_data.iloc[:, 2: -2].to_numpy()
    locations = training_data.iloc[:, -1].to_numpy()
    n_buses = 14

    with open(pinn_file, "r") as fd:
        pinn_ids = json.load(fd)
    with open(arma_file, "r") as fd:
        arma_ids = json.load(fd)

    shared = set(arma_ids).intersection(set(pinn_ids))
    shared = list(shared)
    shared.sort()
    topologies = features[shared, 6 * n_buses:].astype(str)
    print(topologies.shape)
    unique_topologies = np.unique(topologies, axis=0)
    print(unique_topologies.shape)

    # Attack size
    sum_of_sizes = 0
    percents = []
    zeros = []
    for i in shared:
        for bus in range(n_buses):
            angle = features[i, 6 * bus + 4]
            prev_angle = features[i - 1, 6 * bus + 4]
            if prev_angle == 0:
                zeros.append(i)
            if prev_angle != 0:
                attack = angle / prev_angle
                print(attack)
                if attack != 1:
                    percents.append(attack)

            mag = features[i,  6 * bus + 5]
            prev_mag = features[i - 1, 6 * bus + 5]
            if prev_mag != 0:
                attack = mag / prev_mag
                if attack != 1:
                    percents.append(attack)
    print(len(percents))
    ax = plt.subplot()
    ax.hist(percents, bins=20)
    plt.tight_layout()
    plt.savefig("Attack_amounts.png")