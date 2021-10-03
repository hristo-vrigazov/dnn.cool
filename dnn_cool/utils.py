from pathlib import Path
from typing import Union

import joblib
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset, Subset


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
        from torchvision import transforms
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


class RaggedMemoryMapView:

    def __init__(self, ragged_memory_map, subset_selector):
        self.ragged_memory_map = ragged_memory_map
        self.subset_selector = subset_selector

    def __getitem__(self, item):
        indices = self.ragged_memory_map.range[self.subset_selector]
        return RaggedMemoryMapView(self.ragged_memory_map, indices[item])

    def to_list(self):
        indices = self.ragged_memory_map.range[self.subset_selector]
        res = [self.ragged_memory_map[int(idx)] for idx in indices]
        return res


def reduce_shape(shape):
    if isinstance(shape, int):
        return shape
    from operator import mul
    from functools import reduce
    return reduce(mul, shape, 1)


class RaggedMemoryMap:

    def __init__(self, path: Union[str, Path], shapes, dtype, mode,
                 save_metadata=True, initialization_data=None):
        self.path = Path(path)
        path_str = str(self.path)
        self.shapes_path = path_str + '.metadata'
        self.shapes = shapes
        self.dtype = dtype
        if save_metadata:
            self.save_metadata()
        self.mode = mode
        self.flattened_shapes = np.array([reduce_shape(shape) for shape in shapes])
        self.ends = self.flattened_shapes.cumsum()
        self.starts = np.roll(self.ends, 1)
        self.starts[0] = 0
        self.shape = self.flattened_shapes.sum()
        self.n = len(self.shapes)
        self.range = np.arange(self.n)

        self.memmap = np.memmap(path_str, dtype=self.dtype, mode=self.mode, shape=self.shape)
        if initialization_data is not None:
            assert len(initialization_data) == len(shapes), \
                'The initialization data must of the same length as the shapes array!'
            for i in range(len(initialization_data)):
                self[i] = initialization_data[i]

    def save_metadata(self):
        joblib.dump({
            'path': self.path,
            'shapes': self.shapes,
            'dtype': self.dtype
        }, self.shapes_path)

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            indices = self.range[key]
            for i, idx in enumerate(indices):
                self[int(idx)] = value[i]
            return
        if isinstance(key, int):
            self.set_single_index(key, value)
            return
        raise ValueError(f'Unsupported key: {key}')

    def __getitem__(self, item):
        if isinstance(item, slice):
            return RaggedMemoryMapView(self, item)
        if isinstance(item, np.ndarray) and item.dtype == np.bool:
            return RaggedMemoryMapView(self, item)
        if isinstance(item, int):
            return self.get_single_index(item)
        if isinstance(item, np.ndarray) and len(item.shape) <= 0:
            return self.get_single_index(int(item))
        if isinstance(item, np.ndarray) and np.issubdtype(item.dtype, np.integer):
            return RaggedMemoryMapView(self, item)
        if np.isscalar(item):
            return self.get_single_index(int(item))
        raise NotImplementedError(f'Have not yet implemented __getitem__ with {type(item)}.')

    def get_single_index(self, item):
        start = self.starts[item]
        end = self.ends[item]
        shape = self.shapes[item]
        return self.memmap[start:end].reshape(shape)

    def set_single_index(self, key, value):
        start = self.starts[key]
        end = self.ends[key]
        value = np.asarray(value)
        self.memmap[start:end] = value.ravel()

    def __len__(self):
        return self.n

    @classmethod
    def open_existing(cls, path, mode='r+'):
        metadata_path = str(path) + '.metadata'
        metadata = joblib.load(metadata_path)
        return cls(
            path=path,
            shapes=metadata['shapes'],
            dtype=metadata['dtype'],
            mode=mode,
            save_metadata=False,
            initialization_data=None
        )

    @classmethod
    def from_lists_of_int(cls, path, lists_of_int, dtype=np.int64):
        shapes = [len(sample) for sample in lists_of_int]
        return cls(path=path,
                   shapes=shapes,
                   dtype=dtype,
                   mode='w+',
                   initialization_data=lists_of_int)


class StringsMemmap(RaggedMemoryMap):

    def __init__(self, path: Union[str, Path], shapes, dtype=np.int64, mode='r+',
                 save_metadata=True, initialization_data=None):
        super().__init__(path, shapes, dtype, mode=mode,
                         save_metadata=save_metadata,
                         initialization_data=initialization_data)

    @classmethod
    def from_list_of_strings(cls, path, list_of_strings, dtype=np.int64):
        list_of_unicode_ints = []
        shapes = []
        for string in list_of_strings:
            unicode_list = [ord(c) for c in string]
            list_of_unicode_ints.append(unicode_list)
            shapes.append(len(unicode_list))
        return cls(path=path,
                   shapes=shapes,
                   dtype=dtype,
                   mode='w+',
                   initialization_data=list_of_strings)

    def set_single_index(self, key, value):
        super().set_single_index(key, np.array([ord(c) for c in value]))

    def get_single_index(self, item):
        unicode_result = super(StringsMemmap, self).get_single_index(item)
        return ''.join([chr(i) for i in unicode_result])

