import torch
import numpy as np

from dnn_cool.utils import ImageNetNormalizer
from dnn_cool.memmap import RaggedMemoryMap, StringsMemmap


def test_imagenet_normalizer_tensor_input():
    sample = torch.randn(3, 64, 64)
    normalizer = ImageNetNormalizer()
    n = normalizer(sample)
    assert n.dtype == torch.float32
    assert normalizer.should_divide_by_255


def test_imagenet_normalizer_np_input():
    sample = np.random.randint(low=0, high=255, size=(3, 64, 64), dtype=np.uint8)
    normalizer = ImageNetNormalizer()
    n = normalizer(sample)
    assert n.dtype == torch.float32
    assert normalizer.should_divide_by_255


def test_imagenet_normalizer_np_image_input():
    sample = np.random.randint(low=0, high=255, size=(64, 64, 3), dtype=np.uint8)
    normalizer = ImageNetNormalizer()
    n = normalizer(sample)
    assert n.dtype == torch.float32
    assert normalizer.should_divide_by_255


def test_ragged_memory_map_get_single():
    memmap = RaggedMemoryMap('./test_data/ragged.data', [(5,), (6, 2), (3, 4)], dtype=np.uint8, mode='w+')
    expected = np.arange(12).reshape(6, 2)
    memmap[1] = expected
    assert np.all(memmap[1] == expected)


def test_ragged_memory_map_get_slice():
    memmap = RaggedMemoryMap('./test_data/ragged.data', [(5,), (6, 2), (3, 4)], dtype=np.int16, mode='w+')
    expected = [np.arange(12).reshape(6, 2), -np.arange(12).reshape(3, 4)]
    memmap[1:3] = expected
    actual = memmap[1:3].to_list()
    for i in range(2):
        assert np.all(actual[i] == expected[i])


def test_nested_views():
    memmap = RaggedMemoryMap('./test_data/ragged.data', [(5,), (6, 2), (3, 4)], dtype=np.int16, mode='w+')
    expected = [np.arange(12).reshape(6, 2), -np.arange(12).reshape(3, 4)]
    memmap[1:3] = expected
    actual = memmap[1:3][:1].to_list()
    for i in range(1):
        assert np.all(actual[i] == expected[i])


def test_strings_memmap():
    strings = [
        'He is a good developer. That is what is interesting.',
        'She forgot the name, and said торба.',
        'Ще отида до магазина.'
    ]
    memmap = StringsMemmap.from_list_of_strings('./test_data/ragged_str.data', strings)
    assert memmap[1] == strings[1]

    new_memmap = StringsMemmap.open_existing('./test_data/ragged_str.data')
    assert (new_memmap[2] == strings[2])


def test_ragged_equality_zero_shaped():
    memmap = RaggedMemoryMap.from_lists_of_int('./test_data/ragged.data', [[1, 2], [3], [4, 5, 6], [5]])
    r = memmap[np.array(1)]
    assert r is not None

def test_ragged_equality_int_indices():
    memmap = RaggedMemoryMap.from_lists_of_int('./test_data/ragged.data', [[1, 2], [3], [4, 5, 6], [5]])
    r = memmap[np.array([1, 2])]
    assert r is not None
