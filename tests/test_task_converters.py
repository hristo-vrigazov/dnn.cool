import torch

from dnn_cool.task_converters import ToEfficient
from dnn_cool.task_flow import ClassificationTask


def test_efficient_net_head():
    n_classes = 5
    task_name = 'dummy'
    labels = torch.ones(16)
    converter = ToEfficient(ClassificationTask, 'efficientnet-b0', n_classes)
    task = converter(task_name, labels)
    module = task.torch()

    assert module.in_features == 1280
