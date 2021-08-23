from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch
import matplotlib.image as mpimg

from sklearn.preprocessing import MultiLabelBinarizer


def binary_value_converter(values):
    values = values.copy()
    values[np.isnan(values.astype(float))] = -1
    return torch.tensor(values.astype(float)).float().unsqueeze(dim=-1)


def classification_converter(values):
    values = values.copy()
    values[np.isnan(values)] = -1
    return torch.tensor(values).long()


class MultiLabelValuesConverter:

    def __init__(self):
        self.binarizer = MultiLabelBinarizer()
        self.is_fit = False

    def __call__(self, values):
        available_labels = values[~pd.isna(values)]
        labels = []
        for i in range(len(available_labels)):
            split = available_labels.iloc[i].split(',')
            labels.append(list(filter(lambda x: len(x) > 0, split)))
        if not self.is_fit:
            one_hot_labels = self.binarizer.fit_transform(labels)
            self.is_fit = True
        else:
            one_hot_labels = self.binarizer.transform(labels)
        res = np.ones((len(values), one_hot_labels.shape[1]), dtype=np.float32) * -1.
        res[~pd.isna(values)] = one_hot_labels
        return res

    def state_dict(self):
        return {
            'binarizer': self.binarizer,
            'is_fit': self.is_fit
        }

    def load_state_dict(self, state_dict):
        self.binarizer = state_dict['binarizer']
        self.is_fit = state_dict['is_fit']


class ImageCoordinatesValuesConverter:

    def __init__(self, dim: Union[float, np.ndarray]):
        self.dim = dim

    def __call__(self, values):
        values = values.astype(float) / self.dim
        values[np.isnan(values)] = -1
        return torch.tensor(values).float().unsqueeze(dim=-1)


class ImagesFromFileSystem:

    def __init__(self, base_path, values):
        self.base_path = base_path
        self.values = values

    def __getitem__(self, item):
        return mpimg.imread(self.base_path / self.values[item])

    def __len__(self):
        return len(self.values)


class ImagesFromFileSystemConverter:

    def __init__(self, base_path):
        self.base_path = Path(base_path)

    def __call__(self, values):
        return ImagesFromFileSystem(self.base_path, values)
