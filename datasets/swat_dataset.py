import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler


class SWATDataset(Dataset):
    """ Loads a Excel file(s) with physical sensor/actuator readings from a water treatment system"""
    def __init__(self, data_path, sequence_len, window_size):
        self.data = pd.read_excel(data_path, skiprows = 0, header = 1)
        self.sequence_lem = sequence_len
        self.window_size = window_size

        self.scaler = StandardScaler()
        self.features = self.data.iloc[:, 1: -1]
        # self.labels = self.data.iloc[:, -1]
        self.features = self.scaler.fit_transform(self.features)

        self.sequences, self.targets = None, None
        self.make_sequences()

    def make_sequences(self):
        self.sequences = []
        self.targets = []
        for i in range(0, len(self.features) - self.sequence_lem, self.window_size):
            seq = self.features[i:i + self.sequence_lem, :]
            target = self.features[i + self.sequence_lem, :]
            self.sequences.append(seq)
            self.targets.append(target)
        self.sequences = np.array(self.sequences, dtype = np.float32)
        self.targets = np.array(self.targets, dtype = np.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, item):
        return self.sequences[item], self.targets[item]
