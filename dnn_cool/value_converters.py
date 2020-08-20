from typing import Union

import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import MultiLabelBinarizer


def binary_value_converter(values):
    values[np.isnan(values.astype(float))] = -1
    return torch.tensor(values.astype(float)).float().unsqueeze(dim=-1)


def classification_converter(values):
    values[np.isnan(values)] = -1
    return torch.tensor(values).long()


def multilabel_converter(values):
    available_labels = values[~pd.isna(values)]
    labels = []
    for i in range(len(available_labels)):
        split = available_labels.iloc[i].split(',')
        labels.append(list(filter(lambda x: len(x) > 0, split)))
    binarizer = MultiLabelBinarizer()
    one_hot_labels = binarizer.fit_transform(labels)
    res = np.ones((len(values), one_hot_labels.shape[1]), dtype=np.float32) * -1.
    res[~pd.isna(values)] = one_hot_labels
    return res


class ImageCoordinatesValuesConverter:

    def __init__(self, dim: Union[float, np.ndarray]):
        self.dim = dim

    def __call__(self, values):
        values = values.astype(float) / self.dim
        values[np.isnan(values)] = -1
        return torch.tensor(values).float().unsqueeze(dim=-1)
