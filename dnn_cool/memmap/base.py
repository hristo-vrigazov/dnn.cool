from pathlib import Path
from typing import Union

import joblib
import numpy as np
from tqdm import tqdm

from dnn_cool.utils.base import reduce_shape


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


class ShapeView:

    def __init__(self, ragged_shapes, idx=None):
        self.ragged_shapes = ragged_shapes
        self.idx = idx

    def __getitem__(self, item):
        return ShapeView(self.ragged_shapes, item)

    def __eq__(self, other):
        selected = set(self.ragged_shapes if self.idx is None else self.ragged_shapes[self.idx:self.idx+1])
        if len(selected) != 1:
            return False
        return list(selected)[0] == other


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
        self.flattened_shape = self.flattened_shapes.sum()
        self.n = len(self.shapes)
        self.range = np.arange(self.n)

        self.memmap = np.memmap(path_str, dtype=self.dtype, mode=self.mode, shape=self.flattened_shape)
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

    def sum(self):
        return self.memmap.sum()

    @property
    def shape(self):
        return ShapeView(self.shapes)

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

    @classmethod
    def from_stack_of_ndarrays(cls, path, lists_of_ndarray):
        shapes = [ndarray.shape for ndarray in lists_of_ndarray]
        dtype = lists_of_ndarray[0].dtype
        return cls(path=path,
                   shapes=shapes,
                   dtype=dtype,
                   mode='w+',
                   initialization_data=lists_of_ndarray)

    @classmethod
    def from_concat_of_ndarrays(cls, path, lists_of_ndarray):
        dtype = lists_of_ndarray[0].dtype
        if len(lists_of_ndarray[0].shape) == 0:
            return cls(path=path,
                       shapes=[arr.shape for arr in lists_of_ndarray],
                       dtype=dtype,
                       mode='w+',
                       initialization_data=lists_of_ndarray)
        arrs = np.concatenate(lists_of_ndarray, axis=0)
        return cls(path=path,
                   shapes=[arr.shape for arr in arrs],
                   dtype=dtype,
                   mode='w+',
                   initialization_data=arrs)
