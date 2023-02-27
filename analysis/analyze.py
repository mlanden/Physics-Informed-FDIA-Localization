import pickle
from os import path
import json
import pandas as pd
import torch
from torch.utils.data import Subset, DataLoader

from models import prediction_loss, invariant_loss

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


def investigate_invariants(conf, checkpoint, dataset, model):
    invariants_path = path.join(conf["train"]["checkpoint_dir"], conf["train"]["invariants"] + "_invariants.pkl")
    normal_mean_path = path.join(checkpoint, "normal_mean.pt")
    obj = torch.load(normal_mean_path)
    normal_means = obj["mean"]
    normal_stds = obj["std"]

    with open(invariants_path, "rb") as fd:
        invariants = pickle.load(fd)

    with open("analysis/target_ids.json", "r") as fd:
        targets = set(json.load(fd))

    with open("analysis/anomalies.json", "r") as fd:
        ndss_anomalies = json.load(fd)
    with open("analysis/ndss_tps.json", "r") as fd:
        ndss_tps = json.load(fd)

    anomaly_ndss = {i for ids in ndss_anomalies.values() for i in ids}
    anomaly_fns = list(anomaly_ndss.intersection(targets))
    data = Subset(dataset, anomaly_fns)
    loader = DataLoader(data, drop_last=False)
    model = model.cuda()

    for i, (unscaled_seq, scaled_seq, target, attack) in enumerate(loader):
        unscaled_seq = unscaled_seq.cuda()
        scaled_seq = scaled_seq.cuda()
        target = target.cuda()
        outputs, _ = model(unscaled_seq, scaled_seq)

        pred_loss = prediction_loss(unscaled_seq, outputs, target, model.categorical_values)[0]
        
        print(anomaly_fns[i], (pred_loss[4].item() - normal_means[4]) / (normal_stds[4] + 0.01))


if __name__ == '__main__':
    # find_overlap()
    # check_labels()
    investigate_targets()