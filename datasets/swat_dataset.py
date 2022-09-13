import os

import joblib
from os import path
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class SWATDataset(Dataset):
    """ Loads an Excel/csv file(s) with physical sensor/actuator readings from a water treatment system"""

    def __init__(self, conf, data_path, sequence_len, train, load_scaler):
        self.conf = conf

        if data_path.endswith(".xlsx"):
            self.data = pd.read_excel(data_path, skiprows=0, header=1)
        elif data_path.endswith(".csv"):
            self.data = pd.read_csv(data_path, skiprows=0, header=1)

        # for i in self.data:
        #     print(i, self.data[i].unique())

        self.sequence_len = sequence_len
        self.window_size = conf["model"]["window_size"]
        self.checkpoint = conf["train"]["checkpoint"]
        self.train = train
        scale_file = path.join("checkpoint", self.checkpoint, "scaler.gz")

        self.features = self.data.iloc[:, 1: -1].to_numpy()
        self.labels = self.data.iloc[:, -1]

        if load_scaler:
            self.scaler = joblib.load(scale_file)
        else:
            self.scaler = StandardScaler()
            self.scaler.fit(self.features)
            joblib.dump(self.scaler, scale_file)

        if self.train:
            self.sequences, self.targets = None, None
            self.make_sequences()

    def make_sequences(self):
        self.sequences = []
        self.targets = []
        i = 0
        while i < len(self.features) - self.sequence_len:
            seq = self.features[i: i + self.sequence_len, :]
            target = self.features[i + self.sequence_len, :]
            self.sequences.append(seq)
            self.targets.append(target)
            i += self.window_size

        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)
        print(f"Number of sequences: {len(self.sequences)}")

    def __len__(self):
        if self.train:
            return len(self.sequences)
        else:
            return len(self.features) - 1

    def __getitem__(self, item):
        if self.train:
            return self.sequences[item], self.targets[item]
        else:
            return (np.array(self.features[item, :], dtype=np.float32),
                    np.array(self.features[item + 1, :], dtype=np.float32),
                    self.labels.iloc[item] == "Attack")
