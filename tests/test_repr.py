import torch

from dnn_cool.synthetic_dataset import synthetic_dataset_preparation
from dnn_cool.tasks import BinaryClassificationTask
from torch import nn


def test_repr_binary_task():
    task = BinaryClassificationTask(name='is_car', torch_module=nn.Linear(256, 1))
    print(task)


def test_repr_flow():
    model, nested_loaders, datasets, full_flow_for_development, tensorboard_converters = synthetic_dataset_preparation()
    print(full_flow_for_development)

