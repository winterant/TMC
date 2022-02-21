import pickle
import scipy.io
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


def load_toy_example(train=True):
    x, y = pickle.load(open('dataset/ToyExample_2views.pkl', 'rb'))
    num_train = int(len(y) * 0.8)
    if train:
        for v, k in enumerate(x.keys()):
            x[v] = x[k][:num_train]
        y = y[:num_train]
    else:
        for v, k in enumerate(x.keys()):
            x[v] = x[k][num_train:]
        y = y[num_train:]
    for k in x.keys():
        x[k] = MinMaxScaler([0, 1]).fit_transform(x[k]).astype(np.float32)
    return x, y.astype(np.int64)


class MultiViewDataset(Dataset):
    def __init__(self, data_path='dataset/handwritten_6views.mat', train=True):
        super().__init__()
        if data_path == 'toy-example':
            self.x, self.y = load_toy_example(train)
            return
        dataset = scipy.io.loadmat(data_path)
        mode = 'train' if train else 'test'
        num_views = int((len(dataset) - 5) / 2)
        self.x = dict()
        for k in range(num_views):
            view = dataset[f'x{k+1}_{mode}']
            self.x[k] = MinMaxScaler([0, 1]).fit_transform(view).astype(np.float32)
        self.y = dataset[f'gt_{mode}'].flatten()
        if min(self.y) > 0:
            self.y -= 1

    def __getitem__(self, index):
        x = dict()
        for v in self.x.keys():
            x[v] = self.x[v][index]
        return {
            'x': x,
            'y': self.y[index],
            'index': index
        }

    def __len__(self):
        return len(self.y)
