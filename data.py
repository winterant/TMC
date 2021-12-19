import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

"""
You can also download dataset "scene15-gist-hog-lbp.pkl" from "百度网盘":
链接：https://pan.baidu.com/s/1oj8mK5EIencgImd4YmUDTg
提取码：agy7
"""


class MultiViewDataset(Dataset):
    def __init__(self, data_path='dataset/handwritten_6views.pkl', train=True, train_rate=0.8, shuffle=False):
        super().__init__()

        # Load dataset and shuffle the samples
        x, y = pickle.load(open(data_path, 'rb'))
        if shuffle:
            np.random.seed(97)  # Ensure `rand_indices` is always same.
            rand_indices = np.random.permutation(len(y))
            for i in x.keys():
                x[i] = x[i][rand_indices]
            y = y[rand_indices]

        # Normalization.
        for i in x.keys():
            x[i] = MinMaxScaler([0, 1]).fit_transform(x[i]).astype(np.float32)
        y = np.array(y, dtype=np.int64).flatten()

        # Split train and test
        num_train = int(len(y) * train_rate)
        self.x = dict()
        if train:
            for v, k in enumerate(x.keys()):
                self.x[v] = x[k][:num_train]
            self.y = y[:num_train]
        else:
            for v, k in enumerate(x.keys()):
                self.x[v] = x[k][num_train:]
            self.y = y[num_train:]

    def __getitem__(self, index):
        x = dict()
        for v in self.x.keys():
            x[v] = self.x[v][index]
        return x, self.y[index]

    def __len__(self):
        return len(self.y)
