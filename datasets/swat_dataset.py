import os

import joblib
from os import path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from .ics_dataset import ICSDataset


START = 80000
class SWATDataset(ICSDataset):
    """ Loads an Excel/csv file(s) with physical sensor/actuator readings from a water treatment system"""

    def __init__(self, conf, data_path, window_size, train, load_scaler):
        self.conf = conf

        if data_path.endswith(".xlsx"):
            self.data = pd.read_excel(data_path, skiprows=0, header=1)
        elif data_path.endswith(".csv"):
            self.data = pd.read_csv(data_path, skiprows=0, header=1)

        self.sequence_len = conf["model"]["sequence_length"]
        self.window_size = window_size
        self.checkpoint = conf["train"]["checkpoint"]
        self.train = train
        scale_file = path.join("checkpoint", self.checkpoint, "scaler.gz")

        start = START if train else 0
        self.features = self.data.iloc[start:, 1: -1].to_numpy().astype(np.float32)
        self.labels = self.data.iloc[start:, -1] == "Attack"

        if load_scaler and path.exists(scale_file):
            self.scaler = joblib.load(scale_file)
        else:
            self.scaler = MinMaxScaler()
            self.scaler.fit(self.features)
            if path.exists(path.basename(scale_file)):
                joblib.dump(self.scaler, scale_file)
        self.scaled_features = self.scaler.transform(self.features)

        self.sequences, self.targets = None, None
        self.make_sequences()
        print(f"Created {len(self.sequences)} sequences")

    def make_sequences(self):
        self.sequences = []
        self.targets = []
        i = 0
        while i < len(self.features) - self.sequence_len:
            seq = (i, i + self.sequence_len)
            target = i + self.sequence_len
            self.sequences.append(seq)
            self.targets.append(target)
            i += self.window_size

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, item):
        seq_bounds = self.sequences[item]
        unscaled_seq = self.features[seq_bounds[0]: seq_bounds[1]]
        scaled_seq = self.scaled_features[seq_bounds[0]: seq_bounds[1]]
        if self.train:
            return unscaled_seq, scaled_seq, self.features[self.targets[item]]
        else:
            return unscaled_seq, scaled_seq, self.features[self.targets[item]], self.labels.to_numpy()[item]

    def get_data(self):
        return self.features, self.labels

    def get_categorical_features(self):
        # idx of value: number of possibilities
        categorical_values = {2: 3, 3: 2, 4: 2, 9: 3, 10: 2, 11: 2, 12: 2, 13: 2, 14: 2, 15: 2, 19: 3, 20: 3, 21: 3,
                              22: 3, 23: 2, 24: 2, 29: 2, 30: 2, 31: 2, 32: 2, 33: 3, 42: 2, 43: 2, 48: 2, 49: 2, 50: 2}
        return categorical_values

    def get_continuous_features(self):
        features = set(range(51))
        categorical_features = self.get_categorical_features()
        continuous_features = list(features - set(categorical_features.keys()))
        return continuous_features
