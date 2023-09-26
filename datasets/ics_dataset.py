from collections import defaultdict
from os import path
import numpy as np
import pandas as pd
from typing import Tuple
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import joblib


class ICSDataset(ABC, Dataset):

    def __init__(self, conf, window_size, train):
        self.conf = conf
        self.sequence_len = conf["model"]["sequence_length"]
        self.window_size = window_size
        self.checkpoint = conf["train"]["checkpoint"]
        self.predict = conf["model"]["type"] == "prediction"
        self.train = train
        self.scale_file = path.join("checkpoint", self.checkpoint, "scaler.gz")
        self.features = None
        self.labels = None
        self.attack_idxs = None
        self.scaled_features = None
        
    @abstractmethod
    def get_categorical_features(self) -> dict:
        pass

    @abstractmethod
    def get_continuous_features(self):
        pass

    def make_sequences(self, load_scaler):
        if load_scaler and path.exists(self.scale_file):
            self.scaler = joblib.load(self.scale_file)
        else:
            self.scaler = MinMaxScaler()
            self.scaler.fit(self.features)
            if path.exists(path.basename(self.scale_file)):
                joblib.dump(self.scaler, self.scale_file)
        self.scaled_features = self.scaler.transform(self.features)

        self.sequences, self.targets = None, None
        self.sequences = []
        self.targets = []
        self.attack_idxs = []
        attack_map = self.get_attack_map()
        i = 0
        while i < len(self.features) - self.sequence_len:
            seq = (i, i + self.sequence_len)
            if self.predict:
                target = i + self.sequence_len
            else:
                target = i + self.sequence_len - 1
            self.sequences.append(seq)
            self.targets.append(target)
            attack = -1
            for idx in attack_map:
                if i in attack_map[idx]:
                    attack = idx
            self.attack_idxs.append(attack)
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
            return unscaled_seq, scaled_seq, self.features[self.targets[item]], self.labels[item], self.attack_idxs[item]

    def get_data(self):
        return self.features, self.labels

    def get_attack_map(self):
        attack_map = defaultdict(list)
        is_attack = False
        attack_idx = 1
        for i in range(len(self.labels)):
            if self.labels[i]:
                is_attack = True
                attack_map[attack_idx].append(i)

            elif not self.labels[i] and is_attack:
                attack_idx += 1
                is_attack = False
        return attack_map