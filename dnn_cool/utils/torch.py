from pathlib import Path
from typing import Union

import torch
from torch import nn
from torch.utils.data import Subset, Dataset

from dnn_cool.utils.base import split_dataset


def torch_split_dataset(dataset, test_size=0.2, random_state=None):
    train_indices, test_indices = split_dataset(dataset, test_size, random_state=random_state)
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    return train_dataset, test_dataset


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


def load_model_from_export(model, full_flow, out_directory: Union[str, Path]) -> nn.Module:
    out_directory = Path(out_directory)
    model.load_state_dict(torch.load(out_directory / 'state_dict.pth'))
    thresholds_path = out_directory / 'tuned_params.pkl'
    if not thresholds_path.exists():
        return model
    tuned_params = torch.load(thresholds_path)
    full_flow.get_decoder().load_tuned(tuned_params)
    return model


def tensors_to_bool(ll):
    if isinstance(ll, torch.Tensor):
        return ll.bool()
    return [tensors_to_bool(l) for l in ll]


