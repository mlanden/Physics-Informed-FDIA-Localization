
import joblib
from os import path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from .ics_dataset import ICSDataset


class GridDataset(ICSDataset):
    def __init__(self, conf, data_path, window_size, train, load_scaler):
        super().__init__(conf, window_size, train)
        self.data = pd.read_csv(data_path)

        self.features = self.data.iloc[:, 2: -1].to_numpy().astype(np.float32)
        self.labels = self.data.iloc[:, -1] == "Yes"
        self.labels = self.labels.to_numpy()
        self.make_sequences(load_scaler)         

    def get_categorical_features(self):
        return {}
    
    def get_continuous_features(self):
        return list(range(len(self.features[0])))