import pandas as pd
from sklearn.model_selection import train_test_split

class BaseDataLoader:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self):
        raise NotImplementedError

class XGBoostDataLoader(BaseDataLoader):
    def load_data(self):
        from sklearn.datasets import load_diabetes
        data = load_diabetes(as_frame=True)
        X = data.data
        y = data.target
        return train_test_split(X, y, test_size=0.2, random_state=42)

class RFDataLoader(BaseDataLoader):
    def load_data(self):
        from sklearn.datasets import load_diabetes
        data = load_diabetes(as_frame=True)
        X = data.data
        y = data.target
        return train_test_split(X, y, test_size=0.2, random_state=42)
