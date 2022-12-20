# libraries
import pandas as pd
import time
import torch
from torch.utils.data import Dataset


# Dataset creation
class MyDataset(Dataset):
    def __init__(self, path, transform=None):
        df = pd.read_csv(path, index_col=0)
        self.X = df.drop(['target'], axis=1)

        # stats features
        self.X['sum'] = self.X.sum(axis=1)  
        self.X['min'] = self.X.min(axis=1)
        self.X['max'] = self.X.max(axis=1)
        self.X['mean'] = self.X.mean(axis=1)
        self.X['std'] = self.X.std(axis=1)
        self.X['skew'] = self.X.skew(axis=1)
        self.X['kurt'] = self.X.kurtosis(axis=1)
        self.X['med'] = self.X.median(axis=1)
        
        self.X = self.X.values
        self.y = df['target'].values
        self.transform = transform

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        idx = index.tolist()

        x_batch = self.X[index]
        y_batch = self.y[index]
            
        if self.transform is not None:
            x_batch = self.transform(x_batch)
            y_batch = self.transform(y_batch)

        return x_batch, y_batch, idx


# Custom transformation
class to_tens():
    def __call__(self, data):
        return torch.tensor(data, dtype=torch.float32)

    def __repr__(self):
        return self.__class__.__name__ + '()'


def timer(func):
    def wrapper(*args, **kwargs):
        before = time.time()
        dataset = func(*args, **kwargs)
        print(f'Dataset created, took {time.time() - before} sec\n')
        return dataset
    return wrapper

@timer
def create_dataset(path):
    return MyDataset(
        path=path,
        transform=to_tens()
    )
