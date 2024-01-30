from os import path
import numpy as np
import pandas as pd

from .ics_dataset import ICSDataset
from utils import to_complex


class GridDataset(ICSDataset):
    def __init__(self, conf, data_path, train, window_size=1):
        super().__init__(conf, window_size, train)
        self.data = pd.read_csv(data_path)
        self.types = pd.read_csv(conf["data"]["types"])
        self.mva_base = conf["data"]["mva_base"]
        self.n_buses = conf["data"]["n_buses"]
        self.powerworld = conf["data"]["powerworld"]

        self.features = self.data.iloc[:, 2: -2].to_numpy()
        self.labels = self.data.iloc[:, -2] == "Yes"
        self.labels = self.labels.to_numpy()
        self._per_unit()
        self.features = self.features.astype(np.float32)

        self.input_mask = []
        self.output_mask = []
        for bus in range(self.n_buses):
            bus_base_idx = 4 * bus
            bus_type = self.types.iloc[bus, 0]
            if bus_type == 1:
                # PQ
                # reactive
                self.input_mask.append(bus_base_idx)
                # real
                self.input_mask.append(bus_base_idx + 1)
                # voltage angle
                self.output_mask.append(bus_base_idx + 2)
                # voltage mag
                self.output_mask.append(bus_base_idx + 3)
            elif bus_type == 2:
                # PV 
                # real
                self.input_mask.append(bus_base_idx + 1)
                # voltage mag
                self.input_mask.append(bus_base_idx + 3)
                # reactive 
                self.output_mask.append(bus_base_idx)
                # voltage angle
                self.output_mask.append(bus_base_idx + 2)
            elif bus_type == 3:
                # Slack bus
                # voltage angle 
                self.input_mask.append(bus_base_idx + 2)
                # voltage mag
                self.input_mask.append(bus_base_idx + 3)
                # reactive
                self.output_mask.append(bus_base_idx)
                # real
                self.output_mask.append(bus_base_idx + 1)

            else:
                raise RuntimeError("Unknown bus type")
        self.input_mask.extend(list(range(4 * self.n_buses, len(self.features[0]))))
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
        
        for bus in range(self.n_buses):
            bus_idx = 6 * bus
            gen_mvar = self.features[:, bus_idx]
            load_mvar = self.features[:, bus_idx + 2]
            self.features[:, bus_idx] = gen_mvar - load_mvar

            gen_mw = self.features[:, bus_idx + 1]
            load_mw = self.features[:, bus_idx + 3]
            self.features[:, bus_idx + 1] = gen_mw - load_mw

        for bus in reversed(range(self.n_buses)):
            bus_idx = 6 * bus
            self.features = np.delete(self.features, bus_idx + 3, axis=1)
            self.features = np.delete(self.features, bus_idx + 2, axis=1)
        
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
        targets = feat[self.output_mask]
        if self.train:
            return inputs, targets
        else:
            return inputs, targets, self.idx_to_attack[item]
        
    def get_categorical_features(self):
        return {}
    
    def get_continuous_features(self):
        return list(range(len(self.features[0])))
