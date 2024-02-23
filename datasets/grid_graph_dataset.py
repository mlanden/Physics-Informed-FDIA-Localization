import os
from os import path
import json
import numpy as np
import pandas as pd
from typing import List, Tuple
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch.profiler import profile, ProfilerActivity
from tqdm import tqdm

from utils import to_complex
from equations import build_equations

class GridGraphDataset(InMemoryDataset):
    def __init__(self, conf, data_path, pinn_model=None):
        root = data_path[:data_path.index(".csv")]
        if not os.path.exists(root):
            os.makedirs(root)
        
        self.conf = conf        
        self.data_path = data_path
        self.mva_base = conf["data"]["mva_base"]
        self.n_buses = conf["data"]["n_buses"]
        self.powerworld = conf["data"]["powerworld"]
        self.batch_size = conf["train"]["batch_size"]
        self.pinn_model = pinn_model
        if self.pinn_model is not None:
            self.equations = build_equations(conf)
        
        super(InMemoryDataset, self).__init__(root)
        if self.pinn_model is None:
            self.load(self.processed_paths[0])
        else:
            self.load(self.processed_paths[1])

    def __len__(self):
        return len(self.features)
    
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

    @property
    def raw_file_names(self) -> str | List[str] | Tuple:
        return []
    
    @property
    def processed_file_names(self) -> str | List[str] | Tuple:
        return ["data.pt", "transformed.pt"]

    def download(self):
        pass

    def process(self):
        base_data = self.processed_paths[0]
        if not path.exists(base_data):
            self.base_process()
        
        if self.pinn_model is not None:
            self.load(base_data)
            grids = self.embed()
            self.save(grids, self.processed_paths[1])

    def base_process(self):
        self.types = pd.read_csv(self.conf["data"]["types"])
        self.standard_topology = pd.read_csv(self.conf["data"]["ybus"]).map(to_complex)
        data = pd.read_csv(self.data_path).sample(frac=1)
        self.features = data.iloc[:, 2: -2].to_numpy()
        self.locations = data.iloc[:, -1].to_numpy()
        self.labels = data.iloc[:, -2] == "yes"
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

        grids = []
        for graph in tqdm(range(len(self.features))):
            nodes = self.features[graph, self.input_mask][:2 * self.n_buses].reshape(self.n_buses, 2)
            location = self.locations[graph]
            classes = torch.zeros(2 * self.n_buses)
            if location != "-1":
                location = json.loads(location)
                classes.scatter_(0, torch.tensor(location), 1)

            sources = []
            targets = []
            edge_features = []
            i = 0
            j = 0
            for pos in range(4 * self.n_buses, len(self.features[graph]), 2):
                if (self.standard_topology.iloc[i, j].real != 0 or 
                        self.standard_topology.iloc[i, j].imag != 0):
                    sources.append(i)
                    targets.append(j)
                    edge_features.append(self.features[graph, pos: pos + 2])
                j += 1
                if j == self.n_buses:
                    j = 0
                    i += 1
            edge_index = torch.tensor([sources, targets], dtype=torch.long)
            targets = self.features[graph, self.output_mask].reshape(self.n_buses, 2)

            data = Data(x=torch.tensor(nodes, dtype=torch.float),
                        edge_index=edge_index,
                        edge_attr=torch.tensor(np.array(edge_features), dtype=torch.float),
                        y=torch.tensor(targets),
                        classes=classes.view(1, -1)
                        )
            grids.append(data)
            
        self.save(grids, self.processed_paths[0])
    
    @torch.no_grad
    def embed(self):
        loader = DataLoader(self, batch_size=self.batch_size)
        new_grids = []
        for data in tqdm(loader):
            pinn_output = self.pinn_model(data)
            physics_loss = torch.zeros((len(data), len(self.equations)), 
                                    device=pinn_output.device)
            for i, equation in enumerate(self.equations):
                physics_loss[:, i] = equation.confidence_loss(data, pinn_output, None)
            physics_loss = physics_loss.view(-1, 2)
            data.x = torch.hstack((data.x, physics_loss))
            new_grids.extend(data.to_data_list())

        return new_grids
    
    def get_categorical_features(self):
        return {}
    
    def get_continuous_features(self):
        return list(range(len(self.features[0])))