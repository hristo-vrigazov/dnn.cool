import torch
import pytest
from sklearn.metrics import accuracy_score

from torch import nn
from dnn_cool.decoders import BinaryDecoder
from dnn_cool.metrics import Accuracy, NumpyMetric
from dnn_cool.task_flow import Task


@pytest.fixture()
def simple_binary_data():
    x = torch.tensor([-3., 0., 19., -12., 23., 1. -1., -1.]).unsqueeze(dim=-1)
    y = torch.tensor([1, 1, 1, 0, 1, 0, 0])

    task_mock = Task(decoder=BinaryDecoder(), activation=nn.Sigmoid(), labels=None, loss=None, name='mock_task', per_sample_loss=None)
    return x, y, task_mock


def test_binary_accuracy(simple_binary_data):
    x, y, task_mock = simple_binary_data
    metric = Accuracy()
    metric.bind_to_task(task_mock)

    res = metric(x, y, activate=True, decode=True).item()
    assert res >= 0.70


def test_scikit_accuracy(simple_binary_data):
    x, y, task_mock = simple_binary_data
    metric = NumpyMetric(accuracy_score)
    metric.bind_to_task(task_mock)

    res = metric(x, y, activate=True, decode=True).item()
    assert res >= 0.70

