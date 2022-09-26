import pandas as pd
from typing import Tuple
from abc import ABC, abstractmethod
from torch.utils.data import Dataset


class ICSDataset(ABC, Dataset):

    @abstractmethod
    def get_categorical_features(self) -> dict:
        pass

    @abstractmethod
    def get_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        pass
