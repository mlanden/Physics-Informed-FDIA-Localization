import json
from os import path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import torch
from torch.utils.data import Dataset

from utils import to_complex


class GridDataset(Dataset):
    def __init__(self, conf, data_path):
        super().__init__()
        self.n_buses = conf["data"]["n_buses"]
        pt_path = data_path.replace(".csv", ".pt")
        if not path.exists(pt_path):
            data = pd.read_csv(data_path)
            self.mva_base = conf["data"]["mva_base"]
            self.powerworld = conf["data"]["powerworld"]

            self.features = data.iloc[:, 2: -2].to_numpy()
            self.locations = data.iloc[:, -1].to_numpy()
            label_idxs = []
            for i in range(len(self.locations)):
                if str(self.locations[i]) != "-1":
                    label_idxs.append(json.loads(self.locations[i]))
                else:
                    label_idxs.append([])
            mlb = MultiLabelBinarizer()
            
            self.locations = mlb.fit_transform(label_idxs).astype(np.float32)
            self._per_unit()
            self.features = self.features.astype(np.float32)

            data = {"features": self.features,
                    "locations": self.locations}
            torch.save(data, pt_path)
        else:
            data = torch.load(pt_path)
            self.features = data["features"]
            self.locations = data["locations"]
        
        self.input_mask = []
        self.output_mask = []
        for bus in range(self.n_buses):
            bus_base_idx = 4 * bus
            # PQ
            # reactive
            self.input_mask.append(bus_base_idx)
            # real
            self.input_mask.append(bus_base_idx + 1)
            # voltage angle
            self.output_mask.append(bus_base_idx + 2)
            # voltage mag
            self.output_mask.append(bus_base_idx + 3)
        self.input_mask.extend(list(range(4 * self.n_buses, len(self.features[0]))))

    def _per_unit(self):
        for bus in range(self.n_buses):
            if self.powerworld:
                bus_base_idx = 9 * bus
            else:
                bus_base_idx = 8 * bus
            for i in range(6):
                self.features[:, bus_base_idx + i] = self.features[:, bus_base_idx + i] / self.mva_base
            if self.powerworld:
                nominal = self.features[:, bus_base_idx + 7]
                actual = self.features[:, bus_base_idx + 8]
                self.features[:, bus_base_idx + 5] = actual / nominal
        if self.powerworld:
            for bus in reversed(range(self.n_buses)):
                volt_kv_idx = 9 * bus + 8
                self.features = np.delete(self.features, volt_kv_idx, axis=1)
        
        for bus in range(self.n_buses):
            bus_idx = 8 * bus
            gen_mvar = self.features[:, bus_idx]
            load_mvar = self.features[:, bus_idx + 4]
            self.features[:, bus_idx] = gen_mvar - load_mvar

            gen_mw = self.features[:, bus_idx + 2]
            load_mw = self.features[:, bus_idx + 5]
            self.features[:, bus_idx + 2] = gen_mw - load_mw

        for bus in reversed(range(self.n_buses)):
            bus_idx = 8 * bus
            # loads
            self.features = np.delete(self.features, bus_idx + 5, axis=1)
            self.features = np.delete(self.features, bus_idx + 4, axis=1)
            # Base power
            self.features = np.delete(self.features, bus_idx + 3, axis=1)
            self.features = np.delete(self.features, bus_idx + 1, axis=1)
        
        # Y bus
        for i in range(len(self.features)):
            for pos in range(4 * self.n_buses, len(self.features[0])):
                self.features[i, pos] = to_complex(str(self.features[i, pos]))

        extension = np.zeros((len(self.features), self.n_buses ** 2))
        self.features = np.hstack((self.features, extension))
        ybus_base = 4 * self.n_buses
        for i in reversed(range(self.n_buses ** 2)):
            for row in range(len(self.features)):
                ybus = self.features[row, ybus_base + i]
                self.features[row, ybus_base + 2 * i] = ybus.real
                self.features[row, ybus_base + 2 * i + 1] = ybus.imag

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, item):
        feat = self.features[item]
        inputs = feat[self.input_mask]
        targets = self.locations[item]
        return inputs, targets
