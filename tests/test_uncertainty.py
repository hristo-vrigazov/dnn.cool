import torch

from torch import nn
from dnn_cool.uncertainty import DropoutMC


def test_samples_of_correct_shape():
    batch_size = 32
    in_features = 64
    out_features = 16
    module = nn.Linear(in_features, out_features).eval()
    dropout_mc = DropoutMC()

    x = torch.randn((batch_size, in_features))
    actual_shape = (dropout_mc.num_samples,) + module(x).shape
    expected_shape = dropout_mc.create_samples(module, x).shape

    assert actual_shape == expected_shape
