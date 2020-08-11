from functools import partial

import torch

from torch import nn

from dnn_cool.losses import ReducedPerSample


def test_reduced_per_sample_utility(simple_binary_data):
    x, y, task = simple_binary_data

    criterion = nn.BCEWithLogitsLoss(reduction='none')
    expected = criterion(x, y.unsqueeze(dim=1).float())

    x = x.repeat_interleave(3, dim=1).repeat_interleave(4, dim=1)
    y = y.unsqueeze(dim=-1).repeat_interleave(3, dim=1).repeat_interleave(4, dim=1).float()

    criterion = ReducedPerSample(nn.BCEWithLogitsLoss(reduction='none'), torch.mean)
    actual = criterion(x, y)

    assert torch.allclose(expected, actual)
