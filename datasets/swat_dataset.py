import os   
from os import path
import numpy as np
import pandas as pd
from .ics_dataset import ICSDataset


START = 80000
class SWATDataset(ICSDataset):
    """ Loads an Excel/csv file(s) with physical sensor/actuator readings from a water treatment system"""

    def __init__(self, conf, data_path, window_size, train, load_scaler):
        super().__init__(conf, window_size, train)
        if data_path.endswith(".xlsx"):
            self.data = pd.read_excel(data_path, skiprows=0, header=1)
        elif data_path.endswith(".csv"):
            self.data = pd.read_csv(data_path, skiprows=0, header=1)

        start = START if train else 0
        self.features = self.data.iloc[start:, 1: -1].to_numpy().astype(np.float32)
        self.labels = self.data.iloc[start:, -1] == "Attack"
        self.labels = self.labels.to_numpy()

        self.make_sequences(load_scaler)
        print(f"Created {len(self.sequences)} sequences")
        
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
