import torch
import numpy as np

from dnn_cool.utils import ImageNetNormalizer


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
