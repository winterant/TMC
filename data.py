import pickle
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset


class MultiViewToyExample(Dataset):
    def __init__(self, path='dataset/toy_example/ToyExample_2views.pkl', train=None, train_rate=0.8):
        self.x, self.y = pickle.load(open(path, 'rb'))
        self.x = {k: MinMaxScaler().fit_transform(v).astype(np.float32) for k, v in self.x.items()}
        if train is not None:
            train_size = int(len(self.y) * train_rate)
            if train:
                self.x = {k: v[:train_size] for k, v in self.x.items()}
                self.y = np.array(self.y[:train_size], dtype=np.long)
            else:
                self.x = {k: v[train_size:] for k, v in self.x.items()}
                self.y = np.array(self.y[train_size:], dtype=np.long)

    def __getitem__(self, index):
        x = {k: v[index] for k, v in self.x.items()}
        return x, self.y[index]

    def __len__(self):
        return len(self.y)


class HandWrittenDataset(Dataset):
    def __init__(self, data_path='dataset/handwritten_6views.pkl', train=True, train_rate=0.8):
        super().__init__()
        x, y = pickle.load(open(data_path, 'rb'))
        for i in x.keys():
            x[i] = MinMaxScaler([0, 1]).fit_transform(x[i]).astype(np.float32)
        y = np.array(y, dtype=np.long).flatten()

        num_train = int(len(y) * train_rate)
        self.X = dict()
        if train:
            for v, k in enumerate(x.keys()):
                self.X[v] = x[k][:num_train]
            self.Y = y[:num_train]
        else:
            for v, k in enumerate(x.keys()):
                self.X[v] = x[k][num_train:]
            self.Y = y[num_train:]

    def __getitem__(self, index):
        x = dict()
        for v in self.X.keys():
            x[v] = self.X[v][index]
        return x, self.Y[index]

    def __len__(self):
        return len(self.Y)


class Scene15Dataset(Dataset):
    # scene15-gist-hog-lbp.pkl is generated using https://gitee.com/winterantzhao/image-feature-extractor
    def __init__(self, data_path='dataset/scene15-gist-hog-lbp.pkl', train=True):
        super().__init__()
        x, y = pickle.load(open(data_path, 'rb'))
        np.random.seed(97)  # Ensure `rand_indices` is always same.
        rand_indices = np.random.permutation(len(y))
        for i in x.keys():
            x[i] = np.array(x[i])[rand_indices]
            x[i] = MinMaxScaler([0, 1]).fit_transform(x[i]).astype(np.float32)
        y = np.array(y, dtype=np.long).flatten()
        y = y[rand_indices]

        num_train = int(len(y) * 0.8)
        self.X = dict()
        if train:
            for v, k in enumerate(x.keys()):
                self.X[v] = x[k][:num_train]
            self.Y = y[:num_train]
        else:
            for v, k in enumerate(x.keys()):
                self.X[v] = x[k][num_train:]
            self.Y = y[num_train:]

    def __getitem__(self, index):
        x = dict()
        for v in self.X.keys():
            x[v] = self.X[v][index]
        return x, self.Y[index]

    def __len__(self):
        return len(self.Y)
