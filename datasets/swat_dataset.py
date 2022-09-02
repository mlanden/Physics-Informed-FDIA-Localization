import time
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler


class SWATDataset(Dataset):
    """ Loads a Excel file(s) with physical sensor/actuator readings from a water treatment system"""
    def __init__(self, data_path, sequence_len, window_size, read_normal):
        start = time.time()
        if data_path.endswith(".xlsx"):
            self.data = pd.read_excel(data_path, skiprows = 0, header = 1)
        elif data_path.endswith(".csv"):
            self.data = pd.read_csv(data_path, skiprows = 0, header = 1)
        length = time.time() - start
        print(f"Data took {length // 1000} seconds to load")

        self.sequence_len = sequence_len
        self.window_size = window_size
        self.read_normal = read_normal

        self.scaler = StandardScaler()
        self.features = self.data.iloc[:, 1: -1].to_numpy()
        self.labels = self.data.iloc[:, -1]
        self.scaled_features = self.scaler.fit_transform(self.features)

        self.sequences, self.targets = None, None
        self.make_sequences()

    def make_sequences(self):
        self.sequences = []
        self.targets = []
        i = 0
        normal_ids = self.labels.index[self.labels == "Normal"].to_numpy()
        attack_ids = self.labels.index[self.labels == "Attack"].to_numpy()
        print(normal_ids)
        print(attack_ids)
        while i < len(self.features) - self.sequence_len:
            label = self.labels.iloc[i]
            next_attack = np.argmax(attack_ids > i) if len(attack_ids) > 0 else -1
            next_normal = np.argmax(normal_ids > i) if len(normal_ids) > 0 else -1
            if self.read_normal:
                if label == "Normal":
                    start = i
                    end = min(attack_ids[next_attack] - 1, i + self.sequence_len) if next_attack >= 0 else i + self.sequence_len

                    if attack_ids[next_attack] - 1 == i + self.sequence_len:
                        i += self.sequence_len
                else:
                    i = normal_ids[next_normal] if next_normal >= 0 else len(self.features)
                    continue
            else:
                if label == "Attack":
                    start = i
                    end = min(normal_ids[next_normal] - 1, i + self.sequence_len) if next_normal >= 0 else i + self.sequence_len

                    if normal_ids[next_normal] - 1 == i + self.sequence_len:
                        i += self.sequence_len
                else:
                    i = attack_ids[next_attack] if next_attack >= 0 else len(self.features)
                    continue
            print(self.labels.iloc[start:end + 1])
            seq = self.scaled_features[start: end, :]
            target = self.features[end, :]
            self.sequences.append(seq)
            self.targets.append(target)
            i += self.window_size

        self.sequences = np.array(self.sequences, dtype = np.float32)
        self.targets = np.array(self.targets, dtype = np.float32)
        print(f"Number of sequences: {len(self.sequences)}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, item):
        return self.sequences[item], self.targets[item]
