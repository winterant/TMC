import pickle
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset


class MultiViewToyExample(Dataset):

    def __init__(self, views_name=('toy_view1', 'toy_view2'), data_dir='dataset/toy_example', train=True, train_rate=0.8):
        self.Y = pickle.load(open(data_dir + '/toy_labels', 'rb'))
        train_size = int(len(self.Y) * train_rate)
        if train:
            self.Y = np.array(self.Y[:train_size], dtype=np.long)
        else:
            self.Y = np.array(self.Y[train_size:], dtype=np.long)

        self.X = dict()
        for i in range(len(views_name)):
            view = pickle.load(open(data_dir + '/' + views_name[i], 'rb'))
            view = MinMaxScaler([0, 1]).fit_transform(view).astype(np.float32)
            if train:
                view = view[:train_size]
            else:
                view = view[train_size:]
            self.X[i] = view

    def __getitem__(self, index):
        x = dict()
        for v in self.X.keys():
            x[v] = self.X[v][index]
        return x, self.Y[index]

    def __len__(self):
        return len(self.Y)


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
