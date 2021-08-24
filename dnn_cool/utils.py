from pathlib import Path
from typing import Sized, Union

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset, Subset
from torchvision import transforms


def any_value(outputs):
    for key, value in outputs.items():
        if not key.startswith('precondition') and not key == 'gt':
            return value


def torch_split_dataset(dataset, test_size=0.2, random_state=None):
    train_indices, test_indices = split_dataset(dataset, test_size, random_state=random_state)
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    return train_dataset, test_dataset


def split_dataset(n: int, test_size=0.2, random_state=None):
    X, y = np.arange(n), np.arange(n)
    train_indices, test_indices, _, _ = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return train_indices, test_indices


def train_test_val_split(n: int, random_state=None):
    train_indices, val_indices = split_dataset(n, test_size=0.2, random_state=random_state)
    rel_test_indices, rel_val_indices = split_dataset(len(val_indices), test_size=0.5, random_state=random_state)
    return train_indices, val_indices[rel_test_indices], val_indices[rel_val_indices]


class TransformedSubset(Dataset):
    r"""
    Subset of a dataset at specified indices, with additionally applied transforms.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset, indices, sample_transforms=None):
        self.dataset = dataset
        self.indices = indices
        self.transforms = sample_transforms

    def __getitem__(self, idx):
        r = self.dataset[self.indices[idx]]
        if self.transforms is None:
            return r
        return self.transforms(r)

    def __len__(self):
        return len(self.indices)


def to_broadcastable_shape(tensor1, tensor2):
    if len(tensor1.shape) == len(tensor2.shape):
        return tensor1, tensor2
    if len(tensor2.shape) > len(tensor1.shape):
        return to_broadcastable_smaller(tensor1, tensor2)
    tensor2, tensor1 = to_broadcastable_smaller(tensor2, tensor1)
    return tensor1, tensor2


def to_broadcastable_smaller(tensor1, tensor2):
    assert tensor1.shape == tensor2.shape[:len(tensor1.shape)]
    old_axes = tensor1.shape
    new_axes = [1] * (len(tensor2.shape) - len(tensor1.shape))
    return tensor1.view(*old_axes, *new_axes), tensor2


def check_if_should_divide(sample):
    if sample.dtype == np.uint8:
        return True
    if sample.dtype == torch.uint8:
        return True
    if sample.max() > 1.:
        return True
    return False


def check_if_should_transpose(sample):
    return sample.shape[-1] == 3


class ImageNetNormalizer:

    def __init__(self, should_divide_by_255=None, should_transpose=None):
        self.should_divide_by_255 = should_divide_by_255
        self.should_transpose = should_transpose
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def __call__(self, sample):
        if self.should_divide_by_255 is None:
            self.should_divide_by_255 = check_if_should_divide(sample)
        if self.should_transpose is None:
            self.should_transpose = check_if_should_transpose(sample)
        if self.should_divide_by_255:
            sample = sample / 255.
        X = torch.tensor(sample).float()
        if self.should_transpose:
            X = X.permute(2, 0, 1)
        X = self.normalize(X)
        return X


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


def load_model_from_export(model, full_flow, out_directory: Union[str, Path]) -> nn.Module:
    out_directory = Path(out_directory)
    model.load_state_dict(torch.load(out_directory / 'state_dict.pth'))
    thresholds_path = out_directory / 'tuned_params.pkl'
    if not thresholds_path.exists():
        return model
    tuned_params = torch.load(thresholds_path)
    full_flow.get_decoder().load_tuned(tuned_params)
    return model
