import torch
import pytest

from torch import nn
from dnn_cool.decoders import BinaryDecoder
from dnn_cool.metrics import Accuracy
from dnn_cool.task_flow import Task


def test_binary_accuracy():
    x = torch.tensor([-3., 0., 19., -12., 23., 1. -1., -1.]).unsqueeze(dim=-1)
    y = torch.tensor([1, 1, 1, 0, 1, 0, 0])

    task_mock = Task(decoder=BinaryDecoder(), activation=nn.Sigmoid(), labels=None, loss=None, name='mock_task', per_sample_loss=None)
    metric = Accuracy()
    metric.bind_to_task(task_mock)

    res = metric(x, y, activate=True, decode=True).item()
    assert res >= 0.70

