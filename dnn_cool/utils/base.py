from typing import Dict, Sequence

import numpy as np
from sklearn.model_selection import train_test_split


def split_dataset(n: int, test_size=0.2, random_state=None):
    X, y = np.arange(n), np.arange(n)
    train_indices, test_indices, _, _ = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return train_indices, test_indices


def any_value(outputs):
    for key, value in outputs.items():
        if not key.startswith('precondition') and not key == 'gt':
            return value


class Values:

    def __init__(self, keys, values, types):
        assert len(keys) == len(values)
        self.keys = keys
        self.values = values
        self.types = types

    def __getitem__(self, item):
        res = {}
        for i, key in enumerate(self.keys):
            res[key] = self.values[i][item]
        return res

    def __len__(self):
        return len(self.values[0])


def train_test_val_split(n: int, random_state=None):
    train_indices, val_indices = split_dataset(n, test_size=0.2, random_state=random_state)
    rel_test_indices, rel_val_indices = split_dataset(len(val_indices), test_size=0.5, random_state=random_state)
    return train_indices, val_indices[rel_test_indices], val_indices[rel_val_indices]


def reduce_shape(shape):
    if isinstance(shape, int):
        return shape
    from operator import mul
    from functools import reduce
    return reduce(mul, shape, 1)


def create_values_from_dict(inputs: Dict[str, Sequence]):
    keys = list(inputs.keys())
    types = keys.copy()
    values = list(inputs.values())
    return Values(keys=keys, types=types, values=values)


def squeeze_last_axis_if_needed(ndarray):
    if len(ndarray.shape) < 1:
        return ndarray
    if ndarray.shape[-1] == 1:
        return np.squeeze(ndarray, axis=-1)
    return ndarray


def dict_get_along_keys(dct, item):
    if dct is None:
        return dct
    return {k: v[item] for k, v in dct.items()}
