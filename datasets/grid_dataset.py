from os import path
import numpy as np
import pandas as pd

from .ics_dataset import ICSDataset


class GridDataset(ICSDataset):
    def __init__(self, conf, data_path, train, window_size=1):
        super().__init__(conf, window_size, train)
        self.data = pd.read_csv(data_path)
        self.mva_base = conf["data"]["mva_base"]
        self.n_buses = conf["data"]["n_buses"]
        self.powerworld = conf["data"]["powerworld"]

        self.features = self.data.iloc[:, 2: -1].to_numpy().astype(np.float32)
        self.labels = self.data.iloc[:, -1] == "Yes"
        self.labels = self.labels.to_numpy()
        self._per_unit()
        print(np.min(self.features), np.max(self.features))

        self.gen_load_mask = []
        self.voltage_mask = []
        for bus in range(self.n_buses):
            bus_base_idx = 6 * bus
            for i in range(4):
                self.gen_load_mask.append(bus_base_idx + i)
            self.voltage_mask.append(bus_base_idx + 4)
            self.voltage_mask.append(bus_base_idx + 5)
        self.get_attack_map()

    def _per_unit(self):
        for bus in range(self.n_buses):
            if self.powerworld:
                bus_base_idx = 7 * bus
            else:
                bus_base_idx = 6 * bus
            for i in range(4):
                self.features[:, bus_base_idx + i] = (self.features[:, bus_base_idx + i]
                    * 1e6) / (self.mva_base * 1e6)
            if self.powerworld:
                nominal = self.features[:, bus_base_idx + 5]
                actual = self.features[:, bus_base_idx + 6]
                self.features[:, bus_base_idx + 5] = actual / nominal
        if self.powerworld:
            for bus in reversed(range(self.n_buses)):
                volt_kv_idx = 7 * bus + 6
                self.features = np.delete(self.features, volt_kv_idx, axis=1)

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, item):
        feat = self.features[item]
        inputs = feat[self.gen_load_mask]
        targets = feat[self.voltage_mask]
        if self.train:
            return inputs, targets
        else:
            return inputs, targets, self.idx_to_attack[item]
        
    def get_categorical_features(self):
        return {}
    
    def get_continuous_features(self):
        return list(range(len(self.features[0])))