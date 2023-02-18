import json
import pandas as pd


def find_overlap():
    with open("anomalies.json", "r") as fd:
        ndss_anomalies = json.load(fd)
    with open("ndss_tps.json", "r") as fd:
        ndss_tps = json.load(fd)
    with open("fn_ids_physics_informed.json", "r") as fd:
        physics_fns = json.load(fd)

    true_positives_ndss = [i for ids in ndss_anomalies.values() for i in ids]
    true_positives_ndss += [i for ids in ndss_tps.values() for i in ids]

    targets = set(true_positives_ndss).intersection(set(physics_fns))
    print(len(targets))
    with open("target_ids.json", "w") as fd:
        json.dump(list(targets), fd)


def check_labels():
    with open("target_ids.json", "r") as fd:
        targets = json.load(fd)

    data = pd.read_csv("../../data/SWAT/SWaT_Dataset_Attack_modified.csv", skiprows=0, header=1)
    labels = data.iloc[targets, -1]
    assert pd.value_counts(labels)["Attack"] == len(targets)


def investigate_targets():
    with open("target_ids.json", "r") as fd:
        targets = set(json.load(fd))

    with open("anomalies.json", "r") as fd:
        ndss_anomalies = json.load(fd)
    with open("ndss_tps.json", "r") as fd:
        ndss_tps = json.load(fd)
    print("Targets:", len(targets))

    anomaly_ndss = {i for ids in ndss_anomalies.values() for i in ids}
    print("From anomalies:", len(anomaly_ndss.intersection(targets)))

    target_coverage = {}
    for inv in ndss_tps:
        overlap = set(ndss_tps[inv]).intersection(targets)
        if len(overlap) > 0:
            target_coverage[inv] = overlap
    print("Number of invariants:", len(target_coverage))

    target_coverage = dict(sorted(target_coverage.items(), key=lambda x: len(x[1]), reverse=True))
    # anomalies = dict(sorted(self.anomalies.items(), key=lambda x: x[1], reverse=True))
    for i in list(target_coverage.keys())[:10]:
        print(i, len(target_coverage[i]))


if __name__ == '__main__':
    # find_overlap()
    # check_labels()
    investigate_targets()